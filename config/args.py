# encoding:utf-8
# -----------ARGS---------------------
ROOT_DIR = "/home/zyy/pytorch-pretrained-ERNIE/"
DATA_TYPE = "tsv"
RAW_SOURCE_DATA = "data/source_BIO_2014_cropus.txt"
RAW_TARGET_DATA = "data/target_BIO_2014_cropus.txt"
TRAIN_DATA = "data/train.tsv"
VALID_DATA = "data/dev.tsv"
STOP_WORD_LIST = None
CUSTOM_VOCAB_FILE = None
VOCAB_FILE = "ERNIE/vocab.txt"
TRAIN = "data/train.json"
VALID = "data/dev.json"
log_path = "output/logs"
plot_path = "output/images/loss_acc.png"
data_dir = "data/"  # 原始数据文件夹，应包括tsv文件
cache_dir = "model/"
output_dir = "output/checkpoint"  # checkpoint和预测输出文件夹

bert_model = "ERNIE"  # BERT 预训练模型种类 bert-base-chinese
task_name = "bert_ner"  # 训练任务名称

flag_words = ["[PAD]", "[CLP]", "[SEP]", "[UNK]"]
max_seq_length = 200
do_lower_case = True
do_CRF = True
do_not_train_ernie = False
train_batch_size = 16
eval_batch_size = 16
learning_rate = 2e-5
num_train_epochs = 6
warmup_proportion = 0.1
no_cuda = False
seed = 2018
gradient_accumulation_steps = 1
fp16 = False
loss_scale = 0.
if DATA_TYPE == 'txt':
    labels = ["B_PER", "I_PER", "B_T", "I_T", "B_ORG", "I_ORG", "B_LOC", "I_LOC", "O"]
else:
    labels = ["B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "O"]

device = "cuda:0"
