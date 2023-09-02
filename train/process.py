import os.path

from args import args
from prepare import prepare, prepare2
#from test_hubert import test_hubert
from bark_hubert_quantizer.customtokenizer import auto_train

# path = args.path
# mode = args.mode
model = args.hubert_model
#
# if mode == 'prepare':
#     prepare(path)
#
# elif mode == 'prepare2':
#     prepare2(path, model)
#
# elif mode == 'train':
#     auto_train(path, load_model=os.path.join(path, 'model.pth'), save_epochs=args.train_save_epochs)
#
# elif mode == 'test':
#     test_hubert(path, model)

# 生成数据集
# 处理数据集
# 开始训练



def process(mode : str ='' , path : str =''):

    if mode == 'prepare':
        prepare(path)
    elif mode == 'prepare2':
        prepare2(path, model)
    elif mode == 'train':
        auto_train(path, load_model=os.path.join(path, 'model.pth'), save_epochs=args.train_save_epochs)


def process_dataste():
    prepare()


