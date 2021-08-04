from transformers import BertTokenizer, BertForMaskedLM, AutoModelWithLMHead, AutoTokenizer, BertConfig, AutoConfig, \
    XLNetLMHeadModel, DebertaTokenizer, DebertaModel
from pytorch_pretrained_bert.optimization import BertAdam
import torch
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import random, os, sys
import torch.nn as nn
import torch.nn.functional as F
import codecs
import argparse
import spacy
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as color
import matplotlib.pyplot as plt
from nlp import load_dataset
from collections import defaultdict
import json
import csv
import pandas as pd
import seaborn as sns
import random

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

from torch.utils import data
from data_loader import DataProcessor, HalluDataset, get_examples_from_sen_tuple, example2feature
from utils import binary_eval, subsets, sent_ner_bounds
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")




class ClfModel(nn.Module):
      def __init__(self, args):
            super().__init__()

            self.load_model = args.load_model

            if "xlnet" in args.load_model:
                self.tokenizer = AutoTokenizer.from_pretrained(self.load_model)
                self.model = XLNetLMHeadModel.from_pretrained(self.load_model, mem_len=1024).to(args.device)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.load_model)
                config = AutoConfig.from_pretrained(self.load_model)
                config.output_hidden_states = True
                self.model = AutoModelWithLMHead.from_pretrained(self.load_model, config=config).to(args.device)


            hidden_size = 1024 if "large" in self.load_model or self.load_model=="gpt2-medium" else 768

            self.hidden2label = nn.Sequential(
                                    nn.Linear(hidden_size, hidden_size//2),
                                    nn.Sigmoid(),
                                    nn.Linear(hidden_size//2, 2)).to(args.device)

            # self.hidden2label = nn.Linear(hidden_size, 2).to(args.device)
            self.dropout = torch.nn.Dropout(args.dropout)
            self.layer = args.bert_layer

            self.eval()
            self.device = args.device
            self.args = args


      def model_run(self, optim):
            trainpath = os.path.join(self.args.data_path, "train.txt")

            prefix = "runs/{}_lr_{}_dp_{}_{}_clen{}/".format(self.load_model, self.args.lr,
                                            self.args.dropout, self.args.task_mode, self.args.context_len)
            bestmodelpath = prefix + "best_model.pt"
            epoch, epoch_start = self.args.train_epoch, 1
            if os.path.exists(bestmodelpath) and self.args.continue_train:
                checkpoint = torch.load(bestmodelpath)
                self.load_state_dict(checkpoint["model_state_dict"])
                epoch_start = checkpoint["epoch"] + 1

            writer = SummaryWriter(prefix)
            csvlogger = prefix + "valid_log.csv"

            if not os.path.exists(csvlogger):
                csvfile = open(csvlogger, 'w+')
                fileHeader = ["epoch", "H_p", "H_r", "H_f1", "C_p", "C_r", "C_f1", "Gmean",
                              "Acc", "BSS", "ROC_AUC"]
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(fileHeader)
            else:
                csvfile = open(csvlogger, 'a')
                csvwriter = csv.writer(csvfile)

            dp = DataProcessor()
            train_examples = dp.get_examples(trainpath)

            train_dataset = HalluDataset(train_examples, self.tokenizer, self.args.context_len,
                                         self.load_model, self.args.task_mode)

            train_dataloader = data.DataLoader(dataset=train_dataset,
                                               batch_size=self.args.batch_size,
                                               shuffle=True,
                                               num_workers=4,
                                               collate_fn=HalluDataset.pad)
            nSamples = dp.get_label_dist()
            print("====Train label : {}".format(nSamples))
            normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
            normedWeights = torch.FloatTensor(normedWeights).to(self.args.device)
            loss_func = nn.CrossEntropyLoss(weight=normedWeights).to(self.args.device)
            fwd_func = self.model_train
            best_acc, best_f1_score = -1, -1
            for ei in range(epoch_start, epoch+1):
                cnt = 0
                self.train()
                train_loss = 0
                predy, trainy, hallu_sm_score = [], [], []
                for step, batch in enumerate(train_dataloader):
                    batch = tuple(t.to(self.device) for t in batch[:-1])
                    input_ids, input_mask, segment_ids, predict_mask, label_ids = batch
                    score = fwd_func(input_ids, input_mask, segment_ids, predict_mask)
                    hallu_sm = F.softmax(score, dim=1)[:, 1]
                    _, pred = torch.max(score, dim=1)
                    # print("pred {}".format(pred.size()))
                    # print(label_ids.tolist())
                    # print(pred.tolist())
                    trainy.extend(label_ids.tolist())
                    predy.extend(pred.tolist())
                    hallu_sm_score.extend(hallu_sm.tolist())
                    loss = loss_func(score, label_ids)
                    train_loss += loss.item()
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    cnt += 1
                    if cnt % 10 == 0:
                        print("Training Epoch {} - {:.2f}% - Loss : {}".format(ei, 100.0 * cnt/len(train_dataloader), train_loss/cnt))
                print("Training Epoch {} ...".format(ei))
                acc, f1, precision, recall, _, _, _, _ = \
                    binary_eval(predy, trainy, return_f1=True, predscore=hallu_sm_score)
                writer.add_scalar('Loss/train_epoch', train_loss, ei)
                writer.add_scalar('F1/train_consistent_epoch', f1[0], ei)
                writer.add_scalar('Precision/train_consistent_epoch', precision[0], ei)
                writer.add_scalar('Recall/train_consistent_epoch', recall[0], ei)
                writer.add_scalar('F1/train_hallucination_epoch', f1[1], ei)
                writer.add_scalar('Precision/train_hallucination_epoch', precision[1], ei)
                writer.add_scalar('Recall/train_hallucination_epoch', recall[1], ei)
                writer.add_scalar('Acc/train_epoch', acc, ei)
                print("Train Epoch {} end ! Loss : {}".format(ei, train_loss))

                if ei % 4 == 0:
                    savemodel_path = prefix + "model_{}_{}_{}.pt".format(ei, f1[0], f1[1])
                    torch.save(
                    {"model_state_dict": self.state_dict(),
                     "optim_state_dict": optim.state_dict(),
                     "train_f1": f1,
                     "train_precision": precision,
                     "train_recall": recall,
                     "train_acc": acc,
                     "epoch": epoch},
                     savemodel_path)

                validpath = os.path.join(self.args.data_path, "valid.txt")
                valid_examples = dp.get_examples(validpath)
                valid_dataset = HalluDataset(valid_examples, self.tokenizer, self.args.context_len,
                                             self.load_model, self.args.task_mode)
                valid_dataloader = data.DataLoader(dataset=valid_dataset,
                                                   batch_size=self.args.batch_size//2,
                                                   shuffle=False,
                                                   num_workers=4,
                                                   collate_fn=HalluDataset.pad)

                self.eval()
                predy, validy, hallu_sm_score = [], [], []
                valid_loss = 0
                for step, batch in enumerate(valid_dataloader):
                    batch = tuple(t.to(self.device) for t in batch[:-1])
                    input_ids, input_mask, segment_ids, predict_mask, label_ids = batch
                    score = fwd_func(input_ids, input_mask, segment_ids, predict_mask)
                    hallu_sm = F.softmax(score, dim=1)[:, 1]
                    _, pred = torch.max(score, dim=1)
                    validy.extend(label_ids.tolist())
                    predy.extend(pred.tolist())
                    hallu_sm_score.extend(hallu_sm.tolist())
                    loss = loss_func(score, label_ids)
                    valid_loss += loss.item()
                print("Valid Epoch {} ...".format(ei))

                acc, f1, precision, recall, gmean, bss, roc_auc, info = \
                    binary_eval(predy, validy, return_f1=True, predscore=hallu_sm_score)

                if writer:
                    writer.add_scalar('Loss/valid_epoch', valid_loss, ei)
                    writer.add_scalar('F1/valid_consistent_epoch', f1[0], ei)
                    writer.add_scalar('Precision/valid_consistent_epoch', precision[0], ei)
                    writer.add_scalar('Recall/valid_consistent_epoch', recall[0], ei)
                    writer.add_scalar('F1/valid_hallucination_epoch', f1[1], ei)
                    writer.add_scalar('Precision/valid_hallucination_epoch', precision[1], ei)
                    writer.add_scalar('Recall/valid_hallucination_epoch', recall[1], ei)
                    writer.add_scalar('Acc/valid_epoch', acc, ei)

                if csvwriter:
                    rowdata = [ei, precision[1], recall[1], f1[1], precision[0], recall[0], f1[0], gmean, \
                               acc, bss, roc_auc]
                    rowdata = [str(f) for f in rowdata]
                    csvwriter.writerow(rowdata)

                f1_score = f1[0] + f1[1]
                if f1_score > best_f1_score:
                    best_f1_score = f1_score
                    torch.save({"model_state_dict": self.state_dict(),
                                "optim_state_dict": optim.state_dict(),
                                "valid_f1": f1,
                                "valid_precision": precision,
                                "valid_recall": recall,
                                "valid_acc": acc,
                                "epoch": epoch},
                                prefix + "best_model.pt")


      def model_train(self, input_ids, input_mask, segment_ids, predict_mask):

            if "xlnet" in self.load_model:
                _, hidden_states = self.model(input_ids=input_ids, attention_mask=input_mask)
                hidden_states = [h.transpose(0, 1) for h in hidden_states]
            elif "gpt" in self.load_model:
                _, _, hidden_states = self.model(input_ids=input_ids, attention_mask=input_mask)
            else:
                prediction_scores, hidden_states = self.model(input_ids=input_ids, attention_mask=input_mask)

            features = hidden_states[self.layer]
            state = features * predict_mask.unsqueeze(-1)

            maxpool_state = 1.0 * torch.max(state, dim=1)[0]
            maxpool_state = self.dropout(maxpool_state)
            score = self.hidden2label(maxpool_state)

            return score


      def model_eval(self, model_path, data_path):
          dp = DataProcessor()
          testpath = data_path
          test_examples = dp.get_examples(testpath)
          test_dataset = HalluDataset(test_examples, self.tokenizer, self.args.context_len,
                                       self.load_model, self.args.task_mode)
          test_dataloader = data.DataLoader(dataset=test_dataset,
                                             batch_size=self.args.batch_size,
                                             shuffle=False,
                                             num_workers=4,
                                             collate_fn=HalluDataset.pad)

          if os.path.exists(model_path):
              checkpoint = torch.load(model_path)["model_state_dict"]
              model_dict = self.state_dict()
              checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}
              model_dict.update(checkpoint)
              self.load_state_dict(model_dict)
              fwd_func = self.model_train
              predy, testy, hallu_sm_score = [], [], []
              self.eval()
              for step, batch in enumerate(test_dataloader):
                  batch = tuple(t.to(self.device) for t in batch[:-1])
                  input_ids, input_mask, segment_ids, predict_mask, label_ids = batch
                  score = fwd_func(input_ids, input_mask, segment_ids, predict_mask)
                  hallu_sm = F.softmax(score, dim=1)[:, 1]
                  _, pred = torch.max(score, dim=1)
                  testy.extend(label_ids.tolist())
                  predy.extend(pred.tolist())
                  hallu_sm_score.extend(hallu_sm.tolist())
              print("Test ...")

              binary_eval(predy, testy, return_f1=True, predscore=hallu_sm_score)
          else:
              print("Invaild model path ...")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--load_model", default="bert-large-uncased", type=str)
    parser.add_argument("--data_path", default="../data_collections/Wiki-Hades", type=str)
    parser.add_argument("--train_epoch", default=10, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--continue_train", action="store_true")
    parser.add_argument("--task_mode", default="offline", type=str)
    parser.add_argument("--context_len", default=200, type=int) # context length
    parser.add_argument("--bert_layer", default=-1, type=int)
    parser.add_argument("--num_epoch", default=20, type=int)
    parser.add_argument("--params", default="frozen", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--inf_model", default="", type=str)
    parser.add_argument("--inf_data", default="../data_collections/Wiki-Hades/valid.txt", type=str)

    args = parser.parse_args()

    model = ClfModel(args)

    learning_rate0 = args.lr
    weight_decay_finetune = 1e-5

    if "all" in args.params:
        named_params = list(model.hidden2label.named_parameters()) + \
                       list(model.model.named_parameters())
    else:
        named_params = list(model.hidden2label.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay_finetune},
        {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optim_func = torch.optim.Adam if "gpt" in args.load_model else BertAdam
    optimizer = optim_func(optimizer_grouped_parameters, lr=learning_rate0)

    if args.inf_model is "":
        try:
            model.model_run(optimizer)
        except KeyboardInterrupt:
            print("Stop by Ctrl-C ...")
    else:
        model.model_eval(args.inf_model, args.inf_data)

