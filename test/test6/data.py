import random
import re
import string
import jieba
import unicodedata
from config import *

# all_letters = string.ascii_letters + " .,;'"


class Lang:  # language
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # num of words, Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addSentence_cn(self, sentence):
        for word in list(jieba.cut(sentence)):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# 为便于数据处理，把Unicode字符串转换为ASCII编码
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'   # and c in all_letters
    )


# 对英文转换为小写，去空格及非字母符号等处理
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([,])", r" ", s)
    s = re.sub(r"([.!?])", r" \1", s)  # 去除.!?，但是加上\1后又添加了' '和原符号
    # s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # 读文件，然后分成行
    lines = open(DATA_ROOT+'eng-cmn/%s-%s.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')

    # 把行分成语句对，并进行规范化
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # 判断是否需要转换语句对的次序，如[英文，中文]转换为[中文，英文]次序
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

# pairs= [["eng0", "Chi0"],["eng1", "Chi1"],[],[]...]

eng_prefixes = (
    "i am ", "i'm ",
    "he is", "he's ",
    "she is", "she's ",
    "you are", "you're ",
    "we are", "we're ",
    "they are", "they're "
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH \
        # and p[0].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence_cn(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


if __name__ == "__main__":
    input_lang, output_lang, pairs = prepareData('eng', 'cmn')
    print(random.choice(pairs))
    torch.save(input_lang, DATA_ROOT+"eng-cmn/eng.lang")
    torch.save(output_lang, DATA_ROOT + "eng-cmn/cmn.lang")
    torch.save(pairs, DATA_ROOT + "eng-cmn/pairs")
