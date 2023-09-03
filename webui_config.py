import sys


from train.args import args

autolaunch = False
if len(sys.argv) > 1:
    autolaunch = "-autolaunch" in sys.argv
server_name = ''
if len(server_name) < 1:
    server_name = None
server_port = 9999
if server_port <= 0:
    server_port = None
server_share = False

output_folder_path = 'outputs'

app_title = "brark 语音克隆"

#block_theme = 'freddyaboulton/dracula_revamped'
block_theme = 'freddyaboulton/dracula_revamped'
initial_clone_name = "./bark/prompts/custom_prompts/custom_langeuage_en_1"

language_type_list = ["pl", "en", "zh", "ja"]


train_dataset_path = args.train_dataset

train_process_path = args.process

train_path = args.train
