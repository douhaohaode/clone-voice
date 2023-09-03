import json
from bark import text_to_semantic
import os
import random
import numpy
import gradio
from bark.generation import load_model
from bark_hubert_quantizer.customtokenizer import auto_train
from train.prepare import  prepare2
import shutil

output = 'output'
path = os.getcwd() + "/Literature"
#folder_path = os.getcwd() + "/Literature/ready"
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
model_path = os.path.dirname(os.path.abspath(__file__)) + "/models/hubert/hubert.pt"



# 处理数据
def init_prepared(filepath, wav_filepath, progress=gradio.Progress(track_tqdm=True)):
    output_path = filepath + "prepared"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    contents_b = os.listdir(wav_filepath)
    for item in contents_b:
        source_path = os.path.join(wav_filepath, item)
        destination_path = os.path.join(output_path, item)
        shutil.move(source_path, destination_path)

    progress(0, desc="加载模型")
    load_model(use_gpu=True, use_small=False, force_reload=False, model_type='text')

    # 读取 JSON 文件内容
    with open(filepath + "data.json", "r") as file:
        sorted_data = json.load(file)

    for entry in sorted_data:
        progress(1 / len(sorted_data), desc=f"{len(sorted_data) * 100}")
        file_name = entry["file"].replace(".wav", "")
        file_name = file_name + '.npy'
        file_path = os.path.join(output_path , file_name)
        text = entry["text"]
        text = text.strip()
        print(text)
        semantics = text_to_semantic(text, temp=round(random.uniform(0.6, 0.8), ndigits=2))
        numpy.save(file_path, semantics)

    prepare2(filepath, model_path)
    re_name(filepath + 'ready')
    return "完成"


# def sorted_json(json_filepath):
#     with open(json_filepath, "r") as file:
#         data = json.load(file)
#
#         # 筛选出满足条件的元素（以字典形式存储）
#     filtered_data = [item for item in data if
#                      isinstance(item, dict) and "file" in item and item["file"].startswith("e")]
#
#     # 根据 file 字段进行升序排序
#     sorted_data = sorted(filtered_data, key=lambda item: item["file"])
#
#     # 写入新的JSON文件
#     with open("sorted_filtered_data.json", "w") as output_file:
#         json.dump(sorted_data, output_file, indent=4)


# 处理数据
# def process_data():
#     prepare2(path, model_path)
#


#重命名
def re_name(data_path):
    # 获取文件夹中的所有文件名
    file_names = os.listdir(data_path)
    # 按文件名进行降序排序
    sorted_file_names = sorted(file_names, reverse=True)
    # 输出排序后的文件名
    number = 1
    for filename in sorted_file_names:
        print(filename)
        if filename.endswith("_features.npy"):
            comparison_name = filename.replace("_features.npy", "")
            old_path = os.path.join(data_path, filename)
            new_path = os.path.join(data_path, str(number) + "_features.npy")
            os.rename(old_path, new_path)
            for filename1 in sorted_file_names:
                if comparison_name in filename1 and not filename1.endswith("_features.npy"):
                    old_path1 = os.path.join(data_path, filename1)
                    new_path2 = os.path.join(data_path, str(number) + "_semantic.npy")
                    os.rename(old_path1, new_path2)
                    number += 1
                else:
                    print(filename1)
# 重命名
# def rename():
#     # 获取文件夹中的所有文件名
#     file_names = os.listdir(data_path)
#     # 按文件名进行降序排序
#     sorted_file_names = sorted(file_names, reverse=True)
#     for filename in sorted_file_names:
#         if not filename.endswith("_features.npy"):
#             comparison_name = filename.replace(".npy", "")
#             old_path = os.path.join(data_path, filename)
#             new_path = os.path.join(data_path, comparison_name + "_semantic.npy")
#             os.rename(old_path, new_path)
#         else:
#           print(filename)

# 处理数据
# def data_process():
#     prepare2(path, model_path)


# 开始训练
def start_train(path ,save_epochs = 3 , progress=gradio.Progress(track_tqdm=True)):
    progress(0, desc="开始训练")
    auto_train(path, load_model=os.path.join(path, 'model.pth'), save_epochs=save_epochs , progress=gradio.Progress(track_tqdm=True))
    return '完成'

# save_json()
# red_json()
# process_data()
# data_process()
# rename()
#start_train()