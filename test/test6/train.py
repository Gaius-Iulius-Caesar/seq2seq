import random
from torch import optim, nn
from config import *
import jieba
from Seq2SeqModel import EncoderRNN, AttnDecoderRNN
from data import Lang

input_lang = torch.load(DATA_ROOT + "eng-cmn/eng.lang")
output_lang = torch.load(DATA_ROOT + "eng-cmn/cmn.lang")
pairs = torch.load(DATA_ROOT + "eng-cmn/pairs")

teacher_forcing_ratio = 0.5


def tensorFromSentence(lang, sentence):
    indexes = [lang.word2index[word] for word in sentence.split()]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=DEVICE).view(-1, 1)


def tensorFromSentence_cn(lang, sentence):
    indexes = [lang.word2index[word] for word in list(jieba.cut(sentence))]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=DEVICE).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence_cn(output_lang, pair[1])
    return (input_tensor, target_tensor)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=DEVICE)

    loss = torch.tensor([0.0], device=DEVICE)

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]  # tensor中特有写法，即[0][0]

    decoder_input = torch.tensor([[SOS_token]], device=DEVICE)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)  # 输出前1大(最大)的元素及其位置
            decoder_input = topi.squeeze().detach()  # detach from history as input
            # 假设有模型A和模型B，我们需要将A的输出作为B的输入，但训练时我们只训练模型B
            # detach可以使两个计算图的梯度传递断开，从而实现我们所需的功能。
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# import matplotlib.pyplot as plt
# %matplotlib inline

# plt.switch_backend('agg')
# import matplotlib.ticker as ticker
# import numpy as np


# def showPlot(points):
#     plt.figure()
#     fig, ax = plt.subplots()
#     # this locator puts ticks at regular intervals
#     loc = ticker.MultipleLocator(base=0.2)
#     ax.yaxis.set_major_locator(loc)
#     plt.plot(points)


def trainIters(encoder, decoder, n_iters, print_every=1000, learning_rate=0.01):
    start = time.time()
    print_loss_total = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)

        print_loss_total += loss
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))


if __name__ == "__main__":
    hidden_size = 512  # 256
    # encoder = EncoderRNN(input_lang.n_words, EMBEDDING_SIZE, hidden_size).to(DEVICE)
    # attn_decoder = AttnDecoderRNN(EMBEDDING_SIZE, hidden_size, output_lang.n_words, dropout_p=0.1).to(DEVICE)

    encoder = torch.load(MODEL_ROOT + "encoder512.mdl")
    attn_decoder = torch.load(MODEL_ROOT + "attn_decoder512.mdl")

    for i in range(20):
        print("===========", i, "===========")
        trainIters(encoder, attn_decoder, 5000, print_every=1000, learning_rate=LR)

        torch.save(encoder, MODEL_ROOT + "encoder512.mdl")
        torch.save(attn_decoder, MODEL_ROOT + "attn_decoder512.mdl")
