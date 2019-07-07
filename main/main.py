from net.bert_ner import Bert_CRF
from Io.data_loader import create_batch_iter
from train.train import fit
import config.args as args
from util.porgress_util import ProgressBar
from preprocessing.data_processor import produce_data
from util.model_util import save_model, load_model
import argparse

def start():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_not_train_ernie", default=False, action='store_true',)
    parser.add_argument("--do_CRF", default=False, action='store_true', )
    arg=parser.parse_args()
    args.do_not_train_ernie=arg.do_not_train_ernie
    args.do_CRF = arg.do_CRF

    produce_data()
    train_iter, num_train_steps = create_batch_iter("train")
    eval_iter = create_batch_iter("dev")

    epoch_size = num_train_steps * args.train_batch_size * args.gradient_accumulation_steps / args.num_train_epochs

    pbar = ProgressBar(epoch_size=epoch_size, batch_size=args.train_batch_size)
    if args.load_weight:
        model = load_model(args.output_dir)
    else:
        model = Bert_CRF.from_pretrained(args.bert_model,num_tag = len(args.labels))

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    fit(model=model,
        training_iter=train_iter,
        eval_iter=eval_iter,
        num_epoch=args.num_train_epochs,
        pbar=pbar,
        num_train_steps=num_train_steps,
        verbose=1)