import torch.nn as nn
from net.crf import CRF
import numpy as np
from sklearn.metrics import f1_score, classification_report
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from config import args
import torch.nn.functional as F
import torch


class Bert_CRF(BertPreTrainedModel):
    def __init__(self,
                 config,
                 num_tag):
        super(Bert_CRF, self).__init__(config)
        self.bert = BertModel(config)
        if args.do_not_train_ernie:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_tag)
        self.apply(self.init_bert_weights)
        self.crf = CRF(num_tag)
        self.num_tag = num_tag

    def forward(self,
                input_ids,
                token_type_ids,
                attention_mask,
                label_id=None,
                output_all_encoded_layers=False):
        bert_encode, _ = self.bert(input_ids, token_type_ids, attention_mask,
                                   output_all_encoded_layers=output_all_encoded_layers)
        output = self.classifier(bert_encode)

        return output

    def loss_fn(self, bert_encode, output_mask, tags):
        if args.do_CRF:
            loss = self.crf.negative_log_loss(bert_encode, output_mask, tags)
        else:
            loss = torch.autograd.Variable(torch.tensor(0.), requires_grad=True)
            for ix, (features, tag) in enumerate(zip(bert_encode, tags)):
                num_valid = torch.sum(output_mask[ix].detach())
                features = features[output_mask[ix] == 1]
                tag = tag[:num_valid]
                loss_fct = nn.CrossEntropyLoss(ignore_index=0)
                loss = loss + loss_fct(features.view(-1, self.num_tag).cpu(), tag.view(-1).cpu())
        return loss

    def predict(self, bert_encode, output_mask):
        if args.do_CRF:
            predicts = self.crf.get_batch_best_path(bert_encode, output_mask)
            if not args.do_inference:
                predicts = predicts.view(1, -1).squeeze()
                predicts = predicts[predicts != -1]
            else:
                predicts_ =[]
                for ix, features, in enumerate(predicts):
                    #features = features[output_mask[ix] == 1]
                    predict = features[features != -1]
                    predicts_.append(predict)
                predicts = predicts_
        else:
            predicts_ =[]
            for ix, features, in enumerate(bert_encode):
                features = features[output_mask[ix] == 1]
                predict= F.softmax(features,dim=1)
                predict = torch.argmax(predict,dim=1)
                predicts_.append(predict)
            if not args.do_inference:
                predicts=torch.cat(predicts_,0)
            else:
                predicts=predicts_
        return predicts

    def acc_f1(self, y_pred, y_true):
        try:
            y_pred = y_pred.numpy()
            y_true = y_true.numpy()
        except:
            pass
        f1 = f1_score(y_true, y_pred, average="macro")
        correct = np.sum((y_true == y_pred).astype(int))
        acc = correct / y_pred.shape[0]
        return acc, f1

    def class_report(self, y_pred, y_true):
        y_true = y_true.numpy()
        y_pred = y_pred.numpy()
        classify_report = classification_report(y_true, y_pred)
        print('\n\nclassify_report:\n', classify_report)
