
import os
import gradio
import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio
from bark_hubert_quantizer.hubert_manager import HuBERTManager
from bark_hubert_quantizer.pre_kmeans_hubert import CustomHubert
from bark_hubert_quantizer.customtokenizer import CustomTokenizer
from bark.generation import load_codec_model, grab_best_device

# 如果模型无法下载可以打开下面注释下载模型
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
# large_quant_model = True  # Use the larger pretrained model

# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )
device = "cpu"

def clone_voice(audio_filepath, tokenizer_lang, dest_filename, progress=gradio.Progress(track_tqdm=True)):

    use_gpu = not os.environ.get("BARK_FORCE_CPU", False)
    progress(0, desc="Loading Codec")
    model = load_codec_model(use_gpu=use_gpu)

    # From https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer
    hubert_manager = HuBERTManager()
    hubert_manager.make_sure_hubert_installed()
    hubert_manager.make_sure_tokenizer_installed(tokenizer_lang=tokenizer_lang)

    # From https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer
    # Load HuBERT for semantic tokens

    # Load the HuBERT model
    device = grab_best_device(use_gpu)
    hubert_model = CustomHubert(checkpoint_path='./models/hubert/hubert.pt').to(device)

    # Load the CustomTokenizer model
    tokenizer = CustomTokenizer.load_from_checkpoint(f'./models/hubert/{tokenizer_lang}_tokenizer.pth').to(
        device)  # Automatically uses the right layers

    progress(0.25, desc="Converting WAV")

    # Load and pre-process the audio waveform
    wav, sr = torchaudio.load(audio_filepath)
    if wav.shape[0] == 2:  # Stereo to mono if needed
        wav = wav.mean(0, keepdim=True)

    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.to(device)
    progress(0.5, desc="Extracting codes")

    semantic_vectors = hubert_model.forward(wav, input_sample_hz=model.sample_rate)
    semantic_tokens = tokenizer.get_token(semantic_vectors)

    # Extract discrete codes from EnCodec
    with torch.no_grad():
        encoded_frames = model.encode(wav.unsqueeze(0))
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()  # [n_q, T]

    # move codes to cpu
    codes = codes.cpu().numpy()
    # move semantic tokens to cpu
    semantic_tokens = semantic_tokens.cpu().numpy()

    import numpy as np
    output_path = dest_filename + '.npz'
    np.savez(output_path, fine_prompt=codes, coarse_prompt=codes[:2, :], semantic_prompt=semantic_tokens)
    return "Finished"

# def clone_voice(audio_filepath, tokenizer_lang, dest_filename, progress=gradio.Progress(track_tqdm=True)):
#
#     # model = ('quantifier_V1_hubert_base_ls960_23.pth', 'tokenizer_large.pth') if large_quant_model else (
#     #     'quantifier_hubert_base_ls960_14.pth', 'tokenizer.pth')
#     progress(0, desc="Loading HuBERT...")
#
#     # hubert_model = CustomHubert(HuBERTManager.make_sure_hubert_installed(), device=device)
#     hubert_model = CustomHubert(checkpoint_path='./models/hubert/hubert.pt').to(device)
#
#     print('Loading Quantizer...')
#     # quant_model = CustomTokenizer.load_from_checkpoint(HuBERTManager.make_sure_tokenizer_installed(model=model[0], local_file=model[1]), device)
#     quant_model = CustomTokenizer.load_from_checkpoint(f'./models/hubert/{tokenizer_lang}_tokenizer.pth').to(device)
#
#     print('Loading Encodec...')
#     # 使用24kHz的编码器模型
#     encodec_model = EncodecModel.encodec_model_24khz()
#     # encodec_model = EncodecModel.encodec_model_48khz()
#     encodec_model.set_target_bandwidth(6.0)
#     encodec_model.to(device)
#
#
#     print('Downloaded and loaded models!')
#
#    # Put the path to save the cloned speaker to here.
#
#     wav, sr = torchaudio.load(audio_filepath)
#
#     wav_hubert = wav.to(device)
#     if wav_hubert.shape[0] == 2:  # Stereo to mono if needed
#         wav_hubert = wav_hubert.mean(0, keepdim=True)
#
#     print('Extracting semantics...')
#     semantic_vectors = hubert_model.forward(wav_hubert, input_sample_hz=sr)
#     print('Tokenizing semantics...')
#     semantic_tokens = quant_model.get_token(semantic_vectors)
#     print('Creating coarse and fine prompts...')
#     wav = convert_audio(wav, sr, encodec_model.sample_rate, 1).unsqueeze(0)
#
#     wav = wav.to(device)
#
#     with torch.no_grad():
#         encoded_frames = encodec_model.encode(wav)
#     codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()
#
#     codes = codes.cpu()
#     semantic_tokens = semantic_tokens.cpu()
#
#     import numpy as np
#     output_path = dest_filename + '.npz'
#     np.savez(output_path, fine_prompt=codes, coarse_prompt=codes[:2, :], semantic_prompt=semantic_tokens)
#     return "完成"
