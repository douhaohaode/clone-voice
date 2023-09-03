import gradio as gr
import time
import webui_config
from bark_hubert_quantizer.customtokenizer import auto_train
from clone_voice import clone_voice
from datetime import datetime
import numpy as np
from scipy.io.wavfile import write as write_wav
import logging
from transformers import AutoProcessor, BarkModel
import os
from bark.generation import SAMPLE_RATE, preload_models, _load_history_prompt, codec_decode

from util.parseinput import split_and_recombine_text, build_ssml, is_ssml, create_clips_from_ssml
from tqdm.auto import tqdm
from util.helper import create_filename, add_id3_tag
from bark.api import save_as_prompt
from bark.api import generate_with_settings
from xml.sax import saxutils
import torch
import pytorch_seed
from  train_language import  init_prepared, start_train


app_title = webui_config.app_title
selected_theme = webui_config.block_theme
initialname = webui_config.initial_clone_name
autolaunch  = webui_config.autolaunch
server_name  = webui_config.server_name
server_port  = webui_config.server_port
server_share  = webui_config.server_share
tokenizer_language_list = webui_config.language_type_list
output_folder_path = webui_config.output_folder_path

train_dataset_path = webui_config.train_dataset_path
train_dataset_path_wav = webui_config.train_dataset_path

train_process_path = webui_config.train_process_path
train_path = webui_config.train_path


input_text_desired_length = 110
input_text_max_length = 170
silence_sentence = 250
silence_speakers = 500


global run_server
global restart_server

logger = logging.getLogger(__name__)

run_server = True

SAMPLE_RATE = 24_000

bark_models = ["suno/bark-small", "suno/bark"]
selece_model = "suno/bark-small"

def create_filename(path, seed, name, extension):
    now = datetime.now()
    date_str =now.strftime("%m-%d-%Y")
    outputs_folder = os.path.join(os.getcwd(), path)
    if not os.path.exists(outputs_folder):
        os.makedirs(outputs_folder)

    sub_folder = os.path.join(outputs_folder, date_str)
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)

    time_str = now.strftime("%H-%M-%S")
    if seed == None:
        file_name = f"{name}_{time_str}{extension}"
    else:
        file_name = f"{name}_{time_str}_s{seed}{extension}"
    return os.path.join(sub_folder, file_name)

def generate_text_to_speech(text, selected_speaker, selece_model , text_temp, waveform_temp, eos_prob, quick_generation, complete_settings, seed, batchcount, progress=gr.Progress(track_tqdm=True)):
    model_type = "suno/bark"
    if  selece_model == "suno/bark-small" : # use hugingface
        model_type = "suno/bark-small"
        processor = AutoProcessor.from_pretrained(model_type)
        model = BarkModel.from_pretrained(model_type)
        # generation settings
        if selected_speaker == 'None':
            selected_speaker = None

        currentseed = seed
        inputs = processor(text, voice_preset=selected_speaker + ".npz")

        audio_array = model.generate(**inputs)

        audio_array = audio_array.cpu().numpy().squeeze()

        louder_audio_array = audio_array * 10 ** (10 / 20)
        result = create_filename(output_folder_path, currentseed, "bark", ".wav")
        save_wav(louder_audio_array, result)
        return result
    else: # use bark
        # generation settings
        if selected_speaker == 'None':
            selected_speaker = None

        voice_name = selected_speaker

        if text == None or len(text) < 1:
            if selected_speaker == None:
                raise gr.Error('No text entered!')

            # Extract audio data from speaker if no text and speaker selected
            voicedata = _load_history_prompt(voice_name)
            audio_arr = codec_decode(voicedata["fine_prompt"])
            result = create_filename(output_folder_path, "None", "extract", ".wav")
            save_wav(audio_arr, result)
            return result

        if batchcount < 1:
            batchcount = 1

        silenceshort = np.zeros(int((float(silence_sentence) / 1000.0) * SAMPLE_RATE),
                                dtype=np.int16)  # quarter second of silence
        silencelong = np.zeros(int((float(silence_speakers) / 1000.0) * SAMPLE_RATE),
                               dtype=np.float32)  # half a second of silence
        use_last_generation_as_history = "Use last generation as history" in complete_settings
        save_last_generation = "Save generation as Voice" in complete_settings
        for l in range(batchcount):
            currentseed = seed
            if seed != None and seed > 2 ** 32 - 1:
                logger.warning(f"Seed {seed} > 2**32 - 1 (max), setting to random")
                currentseed = None
            if currentseed == None or currentseed <= 0:
                currentseed = np.random.default_rng().integers(1, 2 ** 32 - 1)
            assert (0 < currentseed and currentseed < 2 ** 32)

            progress(0, desc="Generating")

            full_generation = None

            all_parts = []
            complete_text = ""
            text = text.lstrip()
            if is_ssml(text):
                list_speak = create_clips_from_ssml(text)
                prev_speaker = None
                for i, clip in tqdm(enumerate(list_speak), total=len(list_speak)):
                    selected_speaker = clip[0]
                    # Add pause break between speakers
                    if i > 0 and selected_speaker != prev_speaker:
                        all_parts += [silencelong.copy()]
                    prev_speaker = selected_speaker
                    text = clip[1]
                    text = saxutils.unescape(text)
                    if selected_speaker == "None":
                        selected_speaker = None

                    print(
                        f"\nGenerating Text ({i + 1}/{len(list_speak)}) -> {selected_speaker} (Seed {currentseed}):`{text}`")
                    complete_text += text
                    with pytorch_seed.SavedRNG(currentseed):
                        audio_array = generate_with_settings(text_prompt=text, voice_name=selected_speaker,
                                                             semantic_temp=text_temp, coarse_temp=waveform_temp,
                                                             eos_p=eos_prob)
                        currentseed = torch.random.initial_seed()
                    if len(list_speak) > 1:
                        filename = create_filename(output_folder_path, currentseed, "audioclip", ".wav")
                        save_wav(audio_array, filename)
                        add_id3_tag(filename, text, selected_speaker, currentseed)

                    all_parts += [audio_array]
            else:
                texts = split_and_recombine_text(text, input_text_desired_length, input_text_max_length)
                for i, text in tqdm(enumerate(texts), total=len(texts)):
                    print(
                        f"\nGenerating Text ({i + 1}/{len(texts)}) -> {selected_speaker} (Seed {currentseed}):`{text}`")
                    complete_text += text
                    if quick_generation == True:
                        with pytorch_seed.SavedRNG(currentseed):
                            audio_array = generate_with_settings(text_prompt=text, voice_name=selected_speaker,
                                                                 semantic_temp=text_temp, coarse_temp=waveform_temp,
                                                                 eos_p=eos_prob)
                            currentseed = torch.random.initial_seed()
                    else:
                        full_output = use_last_generation_as_history or save_last_generation
                        if full_output:
                            full_generation, audio_array = generate_with_settings(text_prompt=text,
                                                                                  voice_name=voice_name,
                                                                                  semantic_temp=text_temp,
                                                                                  coarse_temp=waveform_temp,
                                                                                  eos_p=eos_prob, output_full=True)
                        else:
                            audio_array = generate_with_settings(text_prompt=text, voice_name=voice_name,
                                                                 semantic_temp=text_temp, coarse_temp=waveform_temp,
                                                                 eos_p=eos_prob)

                    # Noticed this in the HF Demo - convert to 16bit int -32767/32767 - most used audio format
                    # audio_array = (audio_array * 32767).astype(np.int16)

                    if len(texts) > 1:
                        filename = create_filename(output_folder_path, currentseed, "audioclip", ".wav")
                        save_wav(audio_array, filename)
                        add_id3_tag(filename, text, selected_speaker, currentseed)

                    if quick_generation == False and (
                            save_last_generation == True or use_last_generation_as_history == True):
                        # save to npz
                        voice_name = create_filename(output_folder_path, seed, "audioclip", ".npz")
                        save_as_prompt(voice_name, full_generation)
                        if use_last_generation_as_history:
                            selected_speaker = voice_name

                    all_parts += [audio_array]
                    # Add short pause between sentences
                    if text[-1] in "!?.\n" and i > 1:
                        all_parts += [silenceshort.copy()]

            # save & play audio
            result = create_filename(output_folder_path, currentseed, "final", ".wav")
            save_wav(np.concatenate(all_parts), result)
            # write id3 tag with text truncated to 60 chars, as a precaution...
            add_id3_tag(result, complete_text, selected_speaker, currentseed)

        return result

def save_wav(audio_array, filename):
    write_wav(filename, SAMPLE_RATE, audio_array)


def save_voice(filename, semantic_prompt, coarse_prompt, fine_prompt):
    np.savez_compressed(
        filename,
        semantic_prompt=semantic_prompt,
        coarse_prompt=coarse_prompt,
        fine_prompt=fine_prompt
    )


while run_server:
    speakers_list = []
    for root, dirs, files in os.walk("bark/prompts"):
        for file in files:
            if file.endswith(".npz"):
                pathpart = root.replace("./bark/prompts", "")
                name = os.path.join(pathpart, file[:-4])
                if name.startswith("/") or name.startswith("\\"):
                     name = name[1:]
                speakers_list.append(name)

    speakers_list = sorted(speakers_list, key=lambda x: x.lower())
    speakers_list.insert(0, 'None')


    with gr.Blocks(title=f"{app_title}",  mode=f"{app_title}", theme=selected_theme)  as barkgui:
        with gr.Row():
            with gr.Column():
                gr.Markdown(f"### [{app_title}](https://github.com/douhaohaode/clone-voice)")
            # with gr.Column():
            #     gr.HTML(create_version_html(), elem_id="versions")

        with gr.Tab("文字转语音"): # TTS
            with gr.Row():
                with gr.Column():
                    placeholder = "在此输入文字。"  # Enter text here.
                    input_text = gr.Textbox(label="输入文本", lines=4, placeholder=placeholder) # Input Text
                with gr.Column():
                    seedcomponent = gr.Number(label="种子 (default -1 = Random)", precision=0, value=-1)# Seed
                    batchcount = gr.Number(label="批次", precision=0, value=1) #批次数量

            with gr.Row():
                with gr.Column():
                    examples = [
                        "Special meanings: [laughter] [laughs] [sighs] [music] [gasps] [clears throat] MAN: WOMAN:",
                        "♪ Never gonna make you cry, never gonna say goodbye, never gonna tell a lie and hurt you ♪",
                        "And now — a picture of a larch [laughter]",
                        """
                             WOMAN: I would like an oatmilk latte please.
                             MAN: Wow, that's expensive!
                        """,
                    ]
                    examples = gr.Examples(label="例子",examples=examples, inputs=input_text)

                with gr.Column():
                    text_temp = gr.Slider(0.1, 1.0, value=0.6, label="Generation Temperature",
                                          info="1.0 more diverse, 0.1 more conservative")
                    waveform_temp = gr.Slider(0.1, 1.0, value=0.7, label="Waveform temperature",
                                              info="1.0 more diverse, 0.1 more conservative")
            with gr.Row():
                with gr.Column():
                    quick_gen_checkbox = gr.Checkbox(label="Quick Generation", value=True)
                    settings_checkboxes = ["Use last generation as history", "Save generation as Voice"]
                    complete_settings = gr.CheckboxGroup(choices=settings_checkboxes, value=settings_checkboxes,
                                                         label="Detailed Generation Settings", type="value",
                                                         interactive=True, visible=False)
                with gr.Column():
                    eos_prob = gr.Slider(0.0, 0.5, value=0.05, label="End of sentence probability")

            with gr.Row():

                with gr.Column():
                    select_model = gr.Dropdown(bark_models, value=bark_models[1], label="模型选择")  # Voice
                with gr.Column():
                    speaker = gr.Dropdown(speakers_list, value=speakers_list[1], label="音频提示库")
                # with gr.Column():
                #     gr.Markdown("[音频提示库](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c)")  # Voice Prompt Library
            with gr.Row():
                with gr.Column():
                    tts_create_button = gr.Button("生成音频")   # Generate
                with gr.Column():
                    hidden_checkbox = gr.Checkbox(visible=False)
                    button_stop_generation = gr.Button("停止生成") #Stop generation
            with gr.Row():
                output_audio = gr.Audio(label="生成的音频", type="filepath")#Generated Audio

        with gr.Tab("声音克隆"): #Clone vioce
            with gr.Row():
                input_audio_filename = gr.Audio(label="Input audio.wav", source="upload", type="filepath")
            with gr.Row():
                with gr.Column():
                    output_voice = gr.Textbox(label="训练后的语音的文件名", lines=1, placeholder=initialname, value=initialname)
                with gr.Column():
                    tokenizerlang = gr.Dropdown(tokenizer_language_list, label="基础语言分词器", value=tokenizer_language_list[1]) #Base Language Tokenizer
            with gr.Row():
                clone_voice_button = gr.Button("创建音频文件") #Create Voice
            with gr.Row():
                dummy = gr.Text(label="Progress")

        with gr.Tab("训练"):
            with gr.Row():
                with gr.Column():
                    train_dataset_json = gr.Textbox(label="json训练数据集", lines=1, placeholder=train_dataset_path,
                                              value=train_dataset_path)
                    train_dataset_wav = gr.Textbox(label="wav训练数据集", lines=1, placeholder=train_dataset_path_wav,
                                               value=train_dataset_path_wav)
            with gr.Row():
                with gr.Row():
                    train_dataset_button = gr.Button("数据处理")
                with gr.Row():
                    train_dataset_button_stop = gr.Button("停止生成")

            with gr.Row():
                train_dataset_dummy = gr.Text(label="Progress")

            with gr.Row():
                with gr.Column():
                     train_Text = gr.Textbox(label="训练数据路径", lines=1, placeholder=train_path,
                                                value=train_path)
                with gr.Column():
                    epochs = gr.Number(label="纪元", precision=0, value=1)

            with gr.Row():
                with gr.Row():
                    train_button = gr.Button("开始训练")
                with gr.Row():
                    stop_train_button = gr.Button("停止训练")

            with gr.Row():
                 train_dummy = gr.Text(label="Progress")


        gen_click = tts_create_button.click(generate_text_to_speech,
                                            inputs=[input_text, speaker, select_model, text_temp, waveform_temp, eos_prob, quick_gen_checkbox, complete_settings, seedcomponent, batchcount],
                                            outputs=output_audio)
        button_stop_generation.click(fn=None, inputs=None, outputs=None, cancels=[gen_click])


        # TODO clone_test-> clone_voice
        clone_voice_button.click(clone_voice, inputs=[input_audio_filename, tokenizerlang, output_voice], outputs=dummy)

        train_dataset_stop = train_dataset_button.click(init_prepared, inputs=[train_dataset_json, train_dataset_wav], outputs=train_dataset_dummy)

        train_dataset_button_stop.click(fn=None, inputs=None, outputs=None, cancels=[train_dataset_stop])

        train_click = train_button.click(auto_train, inputs=[train_Text, epochs], outputs=train_dummy)

        stop_train_button.click(fn=None, inputs=None, outputs=None, cancels=[train_click])


        restart_server = False
        try:
            barkgui.queue().launch(inbrowser=autolaunch, server_name=server_name, server_port=server_port,
                               share=server_share, prevent_thread_lock=True)
        except:
            restart_server = True
            run_server = False
        try:
            while restart_server == False:
                time.sleep(1.0)
        except (KeyboardInterrupt, OSError):
            print("Keyboard interruption in main thread... closing server.")
            run_server = False
        barkgui.close()

