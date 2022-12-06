# 导入必要的库

from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd
import jieba

import os
import re
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 目的：分词 和 one-hot vector
# 步骤：
# 1 通过类Lang，给每一个word标注唯一的index；
# 2 存储每一个sentence，通过addSentence分词；
# 3 为每一个unknown word创建索引；

SOS_token = 0  # 开始标志
EOS_token = 1  # 结束标志
MAX_LENGTH = 50  # 最大长度


# 初始化类Lang
class Lang:
    def __init__(self):
        # 初始化容器，保存单词和对应的索引
        self.word2index = {}  # 单词 ——> 索引
        self.word2count = {}  # 单词 ——> 频率
        self.index2word = {0: "SOS", 1: "EOS"}  # 索引 ——> 单词
        self.n_words = 2  # 累计，默认值为2 (SOS 和 EOS 两个字符)

    def addSentence(self, sentence):
        # 对语句分词，将每一个单词加入到容器中
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        # 判断单词是否已经在容器中，如果不存在，则添加。同时，统计字符出现的频率
        if word not in self.word2index:
            self.word2index[word] = self.n_words  # 单词对应的索引
            self.word2count[word] = 1  # 单词频率
            self.index2word[self.n_words] = word  # 索引对应的单词
            self.n_words += 1  # 索引加1
        else:
            self.word2count[word] += 1  # 如果单词已经存在，则频率加1


# 目标：Pandas读取、加载数据
# 步骤：
# 1 所有字符串转为小写；
# 2 去掉所有非字符；
# 3 从Unicode转为ASCII；
# 4 分词；

def normalize_sentence(df, lang):
    # 正则化
    # 我们需要将字符转化为ASCII编码
    # 并全部转化为小写字母，并修剪大部分标点符号
    # 除了(a-z, A-Z, ".", "?", "!", ",")这些字符外，全替换成空格
    sentence = df[lang].str.lower()  # 全部转为小写字母
    sentence = sentence.str.replace(r"([.!?])", " \1")  # 将单个字符在A-Z, a-z之间的字符，全部替换为空格
    sentence = sentence.str.replace(r"[^a-zA-Z.!?]+", r" ")  # 将单个字符在A-Z, a-z之间的字符，全部替换为空格
    sentence = sentence.str.normalize('NFD')
    sentence = sentence.str.encode('ascii', errors='ignore').str.decode('utf-8')  # 从Unicode转为ASCII
    return sentence


def cut(sentence):
    words = jieba.cut(sentence, cut_all=False)
    # words = pd.Series(words)[pd.Series(words).apply(len) > 0]  # 去除长度为0的词
    # stopwords = ['：', '“', '！', '”']
    # words = words[~words.isin(stopwords)]
    return ' '.join(words)


def cut_sentence(df, lang):
    for index, row in df.iterrows():
        row[lang] = cut(row[lang])
    return df[lang]


def read_sentence(df, lang1, lang2):
    """ 读取句子 """
    sentence1 = normalize_sentence(df, lang1)  # 调用上面的方法进行预处理
    # sentence2 = normalize_sentence(df, lang2)  # 调用上面的方法进行预处理
    # 中文不进行正则化
    sentence2 = cut_sentence(df, lang2)
    return sentence1, sentence2


def read_file(loc, lang1, lang2):
    """ 读取文件 """
    df = pd.read_csv(loc, delimiter='\t', header=None, names=[lang1, lang2], nrows=2000)
    return df


def process_data(lang1, lang2):
    """ 处理数据 """
    # ！！！注意这里的文件位置，建议用绝对路径，不然读取不到文件，会报错
    df = read_file("D:\\JetBrains\\Python Files\\seq2seq\\cmn_zhsim.txt", lang1, lang2)  # 调用上面的方法read_file
    # print("Read %s sentence pairs" % len(df))
    sentence1, sentence2 = read_sentence(df, lang1, lang2)  # 调用上面的方法read_sentence，即预处理

    # 初始化实例
    source = Lang()
    target = Lang()
    pairs = []
    for i in range(len(df)):
        # 如果文件中，每行的两个字符串的长度，都小于MAX_LENGTH
        if len(sentence1[i].split(' ')) < MAX_LENGTH and len(sentence2[i].split(' ')) < MAX_LENGTH:
            full = [sentence1[i], sentence2[i]]
            source.addSentence(sentence1[i])  # 分词
            target.addSentence(sentence2[i])  # 分词
            pairs.append(full)  # 合并
    return source, target, pairs


# 数据转换为向量Tensor

def indexesFromSentence(lang, sentence):
    """ 获得语句中单词对应的索引 """
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    # 索引
    indexes = indexesFromSentence(lang, sentence)
    # 加入一个结束标志
    # indexes.append(EOS_token)===============================================================
    indexes.append(1)
    # 返回一个tensor类型的索引数据
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorFromPair(input_lang, output_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])  # 输入向量
    target_tensor = tensorFromSentence(output_lang, pair[1])  # 输出向量
    return (input_tensor, target_tensor)


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embbed_dim, num_layers):
        super(Encoder, self).__init__()
        # 设置输入参数
        self.input_dim = input_dim
        self.embbed_dim = embbed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 根据input和embbed的的维度，初始化embedding层
        self.embedding = nn.Embedding(input_dim, self.embbed_dim)
        # 初始化GRU，获取embbed的输入维度，输出隐层的维度，设置GRU层的参数
        self.gru = nn.GRU(self.embbed_dim, self.hidden_dim, num_layers=self.num_layers)

    def forward(self, src):
        embedded = self.embedding(src).view(1, 1, -1)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, embbed_dim, num_layers):
        super(Decoder, self).__init__()

        # 设置参数
        self.embbed_dim = embbed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # 以适当的维度，初始化每一层。
        # decoder层由embedding, GRU, 线性层和Log softmax 激活函数组成
        self.embedding = nn.Embedding(output_dim, self.embbed_dim)
        self.gru = nn.GRU(self.embbed_dim, self.hidden_dim, num_layers=self.num_layers)
        self.out = nn.Linear(self.hidden_dim, output_dim)  # 线性层
        self.softmax = nn.LogSoftmax(dim=1)  # LogSoftmax函数

    def forward(self, input, hidden):
        # reshape the input to (1, batch_size)
        input = input.view(1, -1)
        embedded = F.relu(self.embedding(input))
        output, hidden = self.gru(embedded, hidden)
        prediction = self.softmax(self.out(output[0]))
        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, MAX_LENGTH=MAX_LENGTH):
        super().__init__()
        # 初始化encoder和decoder
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        input_length = source.size(0)  # 获取输入的长度
        batch_size = target.shape[1]
        target_length = target.shape[0]
        vocab_size = self.decoder.output_dim
        outputs = torch.zeros(target_length, batch_size, vocab_size).to(self.device)
        # 为语句中的每一个word编码
        for i in range(input_length):
            encoder_output, encoder_hidden = self.encoder(source[i])
        # 使用encoder的hidden层作为decoder的hidden层
        decoder_hidden = encoder_hidden.to(device)
        # 在预测前，添加一个token
        decoder_input = torch.tensor([SOS_token], device=device)
        # 获取list中的top_k
        # 根据当前的target，预测output word
        for t in range(target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output
            teacher_force = random.random() < teacher_forcing_ratio
            topv, topi = decoder_output.topk(1)
            input = (target[t] if teacher_force else topi)
            if (teacher_force == False and input.item() == EOS_token):
                break
        return outputs


# 将每一个语句对转换为tensor
# 模型采用SGD和NLLLoss
teacher_forcing_ratio = 0.5


def clacModel(model, input_tensor, target_tensor, model_optimizer, criterion):
    model_optimizer.zero_grad()  # 参数初始梯度为0
    input_length = input_tensor.size(0)
    loss = 0
    epoch_loss = 0
    output = model(input_tensor, target_tensor)
    num_iter = output.size(0)

    # 计算损失
    for ot in range(num_iter):
        loss += criterion(output[ot], target_tensor[ot])

    loss.backward()
    model_optimizer.step()
    epoch_loss = loss.item() / num_iter
    return epoch_loss


def trainModel(model, source, target, pairs, num_iteration=20000):
    model.train()  # 训练
    optimizer = optim.SGD(model.parameters(), lr=0.001)  # 梯度
    criterion = nn.NLLLoss()  # 损失
    total_loss_iterations = 0
    training_pairs = [tensorFromPair(source, target, random.choice(pairs))
                      for i in range(num_iteration)]

    for iter in range(1, num_iteration + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        loss = clacModel(model, input_tensor, target_tensor, optimizer, criterion)
        total_loss_iterations += loss
        if iter % 5000 == 0:
            avarage_loss = total_loss_iterations / 5000
            total_loss_iterations = 0
            print("%d %.4f" % (iter, avarage_loss))

    torch.save(model.state_dict(), '../mytraining.pt')  # 保存模型
    return model


def evaluate(model, input_lang, output_lang, sentences, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentences[0])
        output_tensor = tensorFromSentence(output_lang, sentences[1])
        decoded_words = []
        output = model(input_tensor, output_tensor)  # 输出结果

        for ot in range(output.size(0)):
            topv, topi = output[ot].topk(1)
            if topi[0].item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi[0].item()])
    return decoded_words


def evaluateRandomly(model, source, target, pairs, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('source {}'.format(pair[0]))
        print('target {}'.format(pair[1]))
        output_words = evaluate(model, source, target, pair)
        output_sentence = ''.join(output_words)
        print('predicted {}'.format(output_sentence))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    lang1 = 'english'  # 英文
    lang2 = 'chinese'  # 中文
    source, target, pairs = process_data(lang1, lang2)

    randomize = random.choice(pairs)
    # print('random sentence {}'.format(randomize))
    input_size = source.n_words
    output_size = target.n_words
    print('Input: {} Output: {}'.format(input_size, output_size))

    embed_size = 256
    hidden_size = 512
    num_layers = 1
    num_iteration = 100

    encoder = Encoder(input_size, hidden_size, embed_size, num_layers)
    decoder = Decoder(output_size, hidden_size, embed_size, num_layers)

    model = Seq2Seq(encoder, decoder, device).to(device)

    print(encoder)
    print(decoder)

    model = trainModel(model, source, target, pairs, num_iteration)
    evaluateRandomly(model, source, target, pairs)



