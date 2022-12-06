import torch
from torch import nn
import torch.nn.functional as F
from config import *


class EncoderRNN(nn.Module):
    def __init__(self, word_num, input_size, hidden_size):  # 词数，隐藏维数
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(word_num, input_size)  # 词嵌入(num, dim)
        self.gru = nn.GRUCell(input_size, hidden_size)

    def forward(self, input, hidden, batch=1):
        embedded = self.embedding(input).view(batch, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self, batch=1):
        return torch.zeros(batch, self.hidden_size, device=DEVICE)


class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, word_num):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(word_num, input_size)
        self.gru = nn.GRUCell(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, word_num)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, batch=1):
        output = self.embedding(input).view(batch, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output), dim=1)  # batch x hidden -> batch x word_num; softmax at word
        return output, hidden

    def initHidden(self, batch=1):
        return torch.zeros(batch, self.hidden_size, device=DEVICE)


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
        self.attn_combine = nn.Linear(self.hidden_size + self.input_size, self.input_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRUCell(self.input_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs, batch=1):
        embedded = self.embedding(input).view(batch, -1)  # 维数与input一致，故需要进行规范
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded, hidden), 1)), dim=1).unsqueeze(1)
        # batch,input_size+hidden_size ->(attn) batch,max_length -> batch,1,max_length
        attn_applied = torch.bmm(attn_weights, encoder_outputs)  # mm x, mul . ,bmm为3dim对batch的 ×
        # batch,1,max_length x batch,max_length,hidden_size -> batch,hidden_size

        input = torch.cat((embedded, attn_applied), 1)  # batch, input_size+hidden_size
        input = self.attn_combine(input)  # -> batch, hidden_size

        input = F.relu(input)
        hidden = self.gru(input, hidden)

        output = F.log_softmax(self.out(hidden), dim=1)
        return output, hidden

    def initHidden(self, batch=1):
        return torch.zeros(batch, self.hidden_size, device=DEVICE)
