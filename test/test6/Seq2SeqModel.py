import torch
from torch import nn
import torch.nn.functional as F
from config import *


class EncoderRNN(nn.Module):
    def __init__(self, word_num, input_size, hidden_size):  # 词数，隐藏维数
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(word_num, input_size)  # 词嵌入(num, dim)
        self.gru = nn.GRU(input_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)


class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, word_num):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(word_num, input_size)
        self.gru = nn.GRU(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, word_num)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)


class AttnDecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, word_num, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = word_num
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(word_num, input_size)
        self.attn = nn.Linear(self.hidden_size + self.input_size, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size + self.input_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, -1)  # 维数与input一致，故需要进行规范
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded, hidden[0]), 1)), dim=1)
        # 1,input_size+hidden_size ->(attn) 1,max_length
        attn_applied = torch.mm(attn_weights, encoder_outputs)  # mm x, mul . ,bmm为对batch的 ×
        # 1,max_length x max_length,hidden_size -> 1,hidden_size

        output = torch.cat((embedded, attn_applied), 1)  # 1, input_size+hidden_size
        output = self.attn_combine(output)  # ->hidden_size

        output = (F.relu(output)).unsqueeze(0)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)
