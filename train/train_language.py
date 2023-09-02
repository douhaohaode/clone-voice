import json
from bark import text_to_semantic
import os
import random
import numpy

from bark.generation import load_model
from bark_hubert_quantizer.customtokenizer import auto_train
from prepare import prepare2

output = 'output'
path = os.getcwd() + "/Literature"
folder_path = os.getcwd() + "/Literature/ready"
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
model = project_root + "/models/hubert/hubert.pt"


# 读取JSON文件内容
def save_json():
    with open(path + "/set1_transcript.json", "r") as file:
        data = json.load(file)

    # 筛选出满足条件的元素（以字典形式存储）
    filtered_data = [item for item in data if
                     isinstance(item, dict) and "file" in item and item["file"].startswith("e")]

    # 根据 file 字段进行升序排序
    sorted_data = sorted(filtered_data, key=lambda item: item["file"])

    # 写入新的JSON文件
    with open("sorted_filtered_data.json", "w") as output_file:
        json.dump(sorted_data, output_file, indent=4)


def red_json():
    print('Loading semantics model')
    load_model(use_gpu=True, use_small=False, force_reload=False, model_type='text')

    # 读取 JSON 文件内容
    with open("sorted_filtered_data.json", "r") as file:
        sorted_data = json.load(file)

    for entry in sorted_data:
        file_name = entry["file"].replace(".wav", "")
        file_name = file_name + '.npy'
        file_path =  os.path.join(output, file_name)
        text = entry["text"]
        text = text.strip()
        print(text)
        semantics = text_to_semantic(text, temp=round(random.uniform(0.6, 0.8), ndigits=2))
        numpy.save(file_path, semantics)


# 处理数据
def process_data():
    prepare2(path, model)



#处理数据
def data_process():
    # 获取文件夹中的所有文件名
    file_names = os.listdir(folder_path)
    # 按文件名进行降序排序
    sorted_file_names = sorted(file_names, reverse=True)
    # 输出排序后的文件名
    number = 1
    for filename in sorted_file_names:
        print(filename)
        if filename.endswith("_features.npy"):
            comparison_name = filename.replace("_features.npy", "")
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, str(number) + "_features.npy")
            os.rename(old_path, new_path)
            for filename1 in sorted_file_names:
                if comparison_name in filename1 and not filename1.endswith("_features.npy"):
                    old_path1 = os.path.join(folder_path, filename1)
                    new_path2 = os.path.join(folder_path, str(number) + ".npy")
                    os.rename(old_path1, new_path2)
                    number += 1
                else:
                    print(filename1)
# 重命名
def rename():
    # 获取文件夹中的所有文件名
    file_names = os.listdir(folder_path)
    # 按文件名进行降序排序
    sorted_file_names = sorted(file_names, reverse=True)
    for filename in sorted_file_names:
        if not filename.endswith("_features.npy"):
            comparison_name = filename.replace(".npy", "")
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, comparison_name + "_semantic.npy")
            os.rename(old_path, new_path)
        else:
          print(filename)

# 开始训练
def start_train():
    auto_train(path, load_model=os.path.join(path, 'model.pth'), save_epochs=17)

# save_json()
# red_json()
# process_data()
data_process()
# rename()
#start_train()