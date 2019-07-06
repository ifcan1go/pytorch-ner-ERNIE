from Io.data_loader import create_batch_iter
from preprocessing.data_processor import produce_data
import torch
import os
import json
import config.args as args
from util.model_util import load_model

args.do_inference = True
produce_data()

test_iter = create_batch_iter("inference")
epoch_size = args.train_batch_size * args.gradient_accumulation_steps / args.num_train_epochs
model=load_model(args.output_dir)

num_epoch=args.num_train_epochs
device = torch.device(args.device if torch.cuda.is_available() and not args.no_cuda else "cpu")

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

t_total = args.num_train_epochs


if args.fp16:
    try:
        from apex.optimizers import FP16_Optimizer
        from apex.optimizers import FusedAdam
    except ImportError:
        raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

    if args.fp16:
        model.half()

model.to(device)
model.eval()
count = 0
y_predicts, y_labels = [], []
eval_loss, eval_acc, eval_f1 = 0, 0, 0
with torch.no_grad():
    for step, batch in enumerate(test_iter):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, output_mask = batch
        bert_encode = model(input_ids, segment_ids, input_mask).cpu()
        count += 1
        predicts =  model.predict(bert_encode, output_mask)
        y_predicts.append(predicts)
predicts=[]
for i in y_predicts:
    for j in i:
        predicts.append([args.labels[k]for k in j])
with open(os.path.join(args.do_inference_dir,'inference.json'), 'w') as fw:
    for sent in predicts:
        df = {"source": sent}
        encode_json = json.dumps(df)
        print(encode_json, file=fw)

