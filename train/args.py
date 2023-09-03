from argparse import ArgumentParser
import  os


parser = ArgumentParser()

model_path = os.path.join('/models/hubert/', 'hubert.pt')

parser.add_argument('--train_dataset', default='/train/dataset')
parser.add_argument('--process', default='/train/Literature/prepared')
parser.add_argument('--train', default='/train/Literature/ready')

parser.add_argument('--hubert-model', default='./models/hubert/hubert.pt', help='The hubert model to use for preparing the data and later creation of semantic tokens.')
parser.add_argument('--train-save-epochs', default=1, type=int, help='The amount of epochs to train before saving')

args = parser.parse_args()

