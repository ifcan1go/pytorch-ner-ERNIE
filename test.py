from net.bert_ner import Bert_CRF
from Io.data_loader import create_batch_iter
from train.train import fit
import config.args as args
from util.porgress_util import ProgressBar
from preprocessing.data_processor import produce_data

produce_data()

test_iter = create_batch_iter("train")
epoch_size = args.train_batch_size * args.gradient_accumulation_steps / args.num_train_epochs
model = Bert_CRF.from_pretrained(args.bert_model,
          num_tag = len(args.labels))
print (dir(test_iter[0].dataset.__getitem__(0)))
args.labels=1