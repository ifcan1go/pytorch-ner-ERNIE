import os
import json
import random
from collections import Counter
from tqdm import tqdm
from util.Logginger import init_logger
import config.args as args
import operator
import csv
from collections import namedtuple

logger = init_logger("bert_ner", logging_path=args.log_path)

def train_val_split(X, y, valid_size=0.2, random_state=2018, shuffle=True):
    """
    训练集验证集分割
    :param X: sentences
    :param y: labels
    :param random_state: 随机种子
    """
    logger.info('Train val split')

    data = []
    for data_x, data_y in tqdm(zip(X, y), desc='Merge'):
        data.append((data_x, data_y))
    del X, y

    N = len(data)
    test_size = int(N * valid_size)

    if shuffle:
        random.seed(random_state)
        random.shuffle(data)

    valid = data[:test_size]
    train = data[test_size:]

    return train, valid

def sent2char(line):
    """
    句子处理成单词
    :param line: 原始行
    :return: 单词， 标签
    """
    res = line.strip('\n').split()
    return res

def bulid_vocab(vocab_size, min_freq=1, stop_word_list=None):
    """
    建立词典
    :param vocab_size: 词典大小
    :param min_freq: 最小词频限制
    :param stop_list: 停用词 @type：file_path
    :return: vocab
    """
    count = Counter()

    with open(os.path.join(args.ROOT_DIR, args.RAW_SOURCE_DATA), 'r') as fr:
        logger.info('Building vocab')
        for line in tqdm(fr, desc='Build vocab'):
            words, label = sent2char(line)
            count.update(words)

    if stop_word_list:
        stop_list = {}
        with open(os.path.join(args.ROOT_DIR, args.STOP_WORD_LIST), 'r') as fr:
                for i, line in enumerate(fr):
                    word = line.strip('\n')
                    if stop_list.get(word) is None:
                        stop_list[word] = i
        count = {k: v for k, v in count.items() if k not in stop_list}
    count = sorted(count.items(), key=operator.itemgetter(1))
    # 词典
    vocab = [w[0] for w in count if w[1] >= min_freq]
    if vocab_size-3 < len(vocab):
        vocab = vocab[:vocab_size-3]
    vocab = args.flag_words + vocab
    assert vocab[0] == "[PAD]", ("[PAD] is not at the first position of vocab")
    logger.info('Vocab_size is %d'%len(vocab))

    with open(args.VOCAB_FILE, 'w') as fw:
        for w in vocab:
            fw.write(w + '\n')
    logger.info("Vocab.txt write down at {}".format(args.VOCAB_FILE))

def produce_data(custom_vocab=False, stop_word_list=None, vocab_size=None):
    """实际情况下，train和valid通常是需要自己划分的，这里将train和valid数据集划分好写入文件"""
    targets, sentences = [],[]
    with open(os.path.join(args.ROOT_DIR, args.RAW_SOURCE_DATA), 'r') as fr_1, \
            open(os.path.join(args.ROOT_DIR, args.RAW_TARGET_DATA), 'r') as fr_2:
        for sent, target in tqdm(zip(fr_1, fr_2), desc='text_to_id'):
            chars = sent2char(sent)
            label = sent2char(target)

            targets.append(label)
            sentences.append(chars)
            if custom_vocab:
                bulid_vocab(vocab_size, stop_word_list)
    train, valid = train_val_split(sentences, targets)

    with open(args.TRAIN, 'w') as fw:
        for sent, label in train:
            sent = ' '.join([str(w) for w in sent])
            label = ' '.join([str(l) for l in label])
            df = {"source": sent, "target": label}
            encode_json = json.dumps(df)
            print(encode_json, file=fw)
        logger.info('Train set write done')

    with open(args.VALID, 'w') as fw:
        for sent, label in valid:
            sent = ' '.join([str(w) for w in sent])
            label = ' '.join([str(l) for l in label])
            df = {"source": sent, "target": label}
            encode_json = json.dumps(df)
            #print(encode_json, file=fw)
        logger.info('Dev set write done')

produce_data()


def _read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r",encoding='utf-8') as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        headers = next(reader)
        Example = namedtuple('Example', headers)

        examples = []
        for line in reader:
            example = Example(*line)
            examples.append(example)
        return examples

a='data/train.tsv'

ex = _read_tsv(a)
b=os.path.join(args.ROOT_DIR, args.RAW_SOURCE_DATA)
print (a.split('.')[-1])
print (b.split('.')[-1])

with open(args.VALID, 'w') as fw:
    for data in ex:
        sent,label=data.text_a,data.label
        sent = ' '.join([str(w) for w in sent])
        label = ' '.join([str(l) for l in label])
        df = {"source": sent, "target": label}
        encode_json = json.dumps(df)
        print(encode_json, file=fw)
    logger.info('Train set write done')




