import torch

DATA_ROOT = 'D:/JetBrains/Python Files/seq2seq/data/' # the parent root where your train/val/test data are stored
MODEL_ROOT = 'D:/JetBrains/Python Files/seq2seq/models/' # the root to buffer your checkpoints
# LOG_ROOT = '../log/' # the root to log your train/val status
SOS_token = 0
EOS_token = 1

MAX_LENGTH = 20
EMBEDDING_SIZE = 256 # feature dimension
LR = 0.0002 # initial LR
# NUM_EPOCH = 125 # total epoch number (use the firt 1/25 epochs to warm up)
# WEIGHT_DECAY = 5e-4 # do not apply to batch_norm parameters
# MOMENTUM = 0.9
# STAGES = [35, 65, 95] # epoch stages to decay learning rate
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__=="__main__":
    lines = open(DATA_ROOT+'eng-cmn/eng-cmn.txt', encoding='utf-8').read().strip().split('\n')
    print(lines[0])
