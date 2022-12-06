from config import *
from Seq2SeqModel import EncoderRNN, AttnDecoderRNN
from data import Lang, normalizeString

input_lang = torch.load(DATA_ROOT + "eng-cmn/eng.lang")
output_lang = torch.load(DATA_ROOT + "eng-cmn/cmn.lang")
pairs = torch.load(DATA_ROOT + "eng-cmn/pairs")

from train import tensorFromSentence

encoder = torch.load(MODEL_ROOT + "encoder512.mdl")
decoder = torch.load(MODEL_ROOT + "attn_decoder512.mdl")


def predict(sentence):
    sentence = normalizeString(sentence)

    input_tensor = tensorFromSentence(input_lang, sentence)
    input_length = input_tensor.size(0)

    encoder_hidden = encoder.initHidden()
    encoder_outputs = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=DEVICE)

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]  # tensor中特有写法，即[0][0]

    decoder_input = torch.tensor([[SOS_token]], device=DEVICE)
    decoder_hidden = encoder_hidden

    output = []
    for di in range(MAX_LENGTH):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        topv, topi = decoder_output.topk(1)  # 输出前1大(最大)的元素及其位置
        decoder_input = topi.squeeze().detach()
        output.append(topi)
        if decoder_input.item() == EOS_token:
            break
    return output


if __name__ == "__main__":
    while(1):
        sent = input()
        output = predict(sent)
        reply = ""
        for i in output[0:-1]:
            reply += output_lang.index2word[i.item()]
        print(reply)
