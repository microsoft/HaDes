from transformers import BertTokenizer, BertForMaskedLM, AutoModelWithLMHead, AutoTokenizer, BertConfig, AutoConfig
from pytorch_pretrained_bert.optimization import BertAdam
import torch
from nltk import sent_tokenize
from nltk.corpus import stopwords
import random, os
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
import pandas as pd
import seaborn as sns
import random

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, hamming_loss, \
    f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings("ignore")
from utils import binary_eval, remove_marked_sen


def get_ppmi_matrix(voc, path="../data_collections/Wiki-Hades/train.txt"):
    def co_occurrence(sentences, window_size):
        d = defaultdict(int)
        vocab = set(voc)
        for text in sentences:
            # iterate over sentences
            # print(text)
            for i in range(len(text)):
                token = text[i]
                next_token = text[i+1 : i+1+window_size]
                for t in next_token:
                    if t in vocab and token in vocab:
                        key = tuple( sorted([t, token]) )
                        d[key] += 1
        # print(vocab)
        print(len(vocab))

        # formulate the dictionary into dataframe
        vocab = sorted(vocab) # sort vocab
        df = pd.DataFrame(data=np.zeros((len(vocab), len(vocab)), dtype=np.int16),
                          index=vocab,
                          columns=vocab)
        for key, value in d.items():
            df.at[key[0], key[1]] = value
            df.at[key[1], key[0]] = value
        return df

    def pmi(df, positive=True):
        col_totals = df.sum(axis=0)
        total = col_totals.sum()
        row_totals = df.sum(axis=1)
        expected = np.outer(row_totals, col_totals) / total
        df = df / expected
        return df

    corpus = []

    with codecs.open(path, "r", encoding="utf-8") as fr:
        for line in fr:
            example = json.loads(line.strip())
            tgt, tgt_ids = example["replaced"], example["replaced_ids"]
            sen = remove_marked_sen(tgt, tgt_ids[0], tgt_ids[1])
            corpus.append(sen)

    df = co_occurrence(corpus, 1)
    ppmi = pmi(df, positive=True)
    print("finish")
    return ppmi


def get_idf_matrix(path="../data_collections/Wiki-Hades/train.txt"):
    corpus = []

    with codecs.open(path, "r", encoding="utf-8") as fr:
        for line in fr:
            example = json.loads(line.strip())
            tgt, tgt_ids = example["replaced"], example["replaced_ids"]
            sen = remove_marked_sen(tgt, tgt_ids[0], tgt_ids[1])
            corpus.append(" ".join(sen))

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus).toarray()
    word = vectorizer.get_feature_names()
    num_doc, num_vocab = X.shape
    X = np.array(X>0, dtype=int)
    word_idf = np.log10(num_doc / (X.sum(0)+1))
    idf_dic = dict()
    for w, idf in zip(word, word_idf):
        idf_dic[w] = idf

    word_freq = X.sum(0)
    word_freq = word_freq / word_freq.sum()

    return idf_dic, word_freq


def subsets(nums):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    ans = []
    def dfs(curpos, tmp):
        if tmp:
            ans.append(tmp[:])
        for i in range(curpos, len(nums)):
            tmp.append(nums[i])
            dfs(i+1, tmp)
            tmp.pop(-1)
    dfs(0, [])
    return ans


class ClfModel:
      def __init__(self, args):
            self.idf_dic, self.p_word = get_idf_matrix()
            if not os.path.exists("ppmi.pkl"):
                print("reading ppmi ...")
                word_ppmi = get_ppmi_matrix(list(self.idf_dic.keys())[:])
                self.word_ppmi = word_ppmi
                word_ppmi.to_pickle("ppmi.pkl")
            else:
                word_ppmi = pd.read_pickle("ppmi.pkl")
                self.word_ppmi = word_ppmi
            self.args = args
            self.device = args.device
            self.model = args.model
            self.rep_model = AutoModelWithLMHead.from_pretrained("bert-base-uncased").to(self.device)
            self.rep_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            if "svm" in self.model:
                self.clf = svm.LinearSVC()
            else:
                self.clf = make_pipeline(StandardScaler(),
                                         SGDClassifier(loss="log", max_iter=10000, tol=1e-5))

      def get_ppmi_features(self, rep_sen, rep_ids):
            rep_start_id, rep_end_id = rep_ids
            rep_tokens = remove_marked_sen(rep_sen, rep_start_id, rep_end_id)

            max_ppmi, mean_ppmi, min_ppmi = [], [], []
            for idx in range(rep_start_id, rep_end_id+1):
                ppmis = []
                for j in range(rep_start_id):
                    v = 0
                    if rep_tokens[idx] in self.word_ppmi.columns and \
                       rep_tokens[j] in self.word_ppmi.columns:
                        v = self.word_ppmi.at[rep_tokens[idx], rep_tokens[j]]
                        if v > 0:
                            v = max(0, np.log(v))
                        else:
                            v = 0 # not gona happen
                    ppmis.append(v)
                max_ppmi.append(max(ppmis))
                min_ppmi.append(min(ppmis))
                mean_ppmi.append(sum(ppmis)/len(ppmis))

            return sum(mean_ppmi) / len(mean_ppmi), sum(max_ppmi) / len(max_ppmi), sum(min_ppmi) / len(min_ppmi)

      def get_tfidf_features(self, rep_sen, rep_ids):
            rep_start_id, rep_end_id = rep_ids
            rep_tokens = remove_marked_sen(rep_sen, rep_start_id, rep_end_id)

            #  TF-IDF features
            tf_dic = dict()
            for token in rep_tokens:
                if token in self.idf_dic:
                    if token not in tf_dic:
                        tf_dic[token] = 1
                    else:
                        tf_dic[token] += 1

            tf_total = sum(tf_dic.values())
            tfidf_list = []
            for idx in range(rep_start_id, rep_end_id+1):
                if rep_tokens[idx] in self.idf_dic:
                    tfidf_list.append(tf_dic[rep_tokens[idx]] * self.idf_dic[rep_tokens[idx]] / tf_total)
            tfidf_max = max(tfidf_list) if tfidf_list else 0.
            tfidf_min = min(tfidf_list) if tfidf_list else 0.
            tfidf_mean = sum(tfidf_list)/len(tfidf_list) if tfidf_list else 0.
            return tfidf_mean, tfidf_max, tfidf_min

      def encode_bert(self, rep_sen, rep_ids):
            rep_start_id, rep_end_id = rep_ids
            rep_tokens = remove_marked_sen(rep_sen, rep_start_id, rep_end_id)
            #  Prob, Entropy features
            rep_subtokens = ["[CLS]"]
            tokenizer = self.rep_tokenizer
            model = self.rep_model
            rep_mask_start_id, rep_mask_end_id = 0, 0
            for id, rep_token in enumerate(rep_tokens):
                rep_subtoken = tokenizer.tokenize(rep_token)
                if id == rep_start_id:
                    rep_mask_start_id = len(rep_subtokens)
                if id == rep_end_id:
                    rep_mask_end_id = len(rep_subtokens) + len(rep_subtoken)
                if id >= rep_start_id and id <= rep_end_id:
                    rep_subtokens.extend(len(rep_subtoken) * ["[MASK]"])
                else:
                    rep_subtokens.extend(rep_subtoken)
            rep_subtokens.append("[SEP]")
            rep_input_ids = torch.LongTensor(tokenizer.convert_tokens_to_ids(rep_subtokens)).unsqueeze(0).to(self.device)
            prediction_scores = model(rep_input_ids)[0]
            prediction_scores = F.softmax(prediction_scores, dim=-1)

            scores = []
            for id in range(rep_mask_start_id, rep_mask_end_id):
                subtoken_score = prediction_scores[0, id, rep_input_ids[0][id]].item()
                scores.append(subtoken_score)

            entropies = []
            for id in range(rep_mask_start_id, rep_mask_end_id):
                vocab_scores = prediction_scores[0, id].detach().cpu().numpy()
                entropy = np.sum(np.log(vocab_scores+1e-11) * vocab_scores)
                entropies.append(-entropy)

            return sum(scores)/len(scores), max(scores), min(scores), \
                   sum(entropies)/len(entropies), max(entropies), min(entropies)



      def train(self, trainpath="../data_collections/Wiki-Hades/train.txt",
              testpath="../data_collections/Wiki-Hades/valid.txt", epoch=10):
            feature_names = {"avgscore": 0, "avgentro": 1, "avgtfidf": 2, "avgppmi": 3,
                             "maxscore": 4, "maxentro": 5, "maxtfidf": 6, "maxppmi": 7}
            feature_keys = list(feature_names.keys())[:]
            combinations = subsets(feature_keys)

            encode_func = self.encode_bert
            trainx, trainy = [], []
            print("Load Training Features ...")
            with codecs.open(trainpath, "r", encoding="utf-8") as fr:
                cnt = 0
                for line in fr:
                    example = json.loads(line.strip())
                    label = example["hallucination"]
                    if label == 2: continue
                    avgscore, maxscore, _, avgentro, maxentro, _ = encode_func(example["replaced"], example["replaced_ids"])
                    avgtfidf, maxtfidf, _ = self.get_tfidf_features(example["replaced"], example["replaced_ids"])
                    avgppmi, maxppmi, _ = self.get_ppmi_features(example["replaced"], example["replaced_ids"])
                    features = [avgscore, avgentro, avgtfidf, avgppmi,
                                maxscore, maxentro, maxtfidf, maxppmi]

                    trainx.append(features)
                    trainy.append(label)
                    cnt += 1
                    if cnt % 500 == 0:
                        print("Train {}".format(cnt))
            print(trainx[:30])

            testx, testy = [], []
            with codecs.open(testpath, "r", encoding="utf-8") as fr:
                cnt = 0
                for line in fr:
                    example = json.loads(line.strip())
                    label = example["hallucination"]
                    if label == 2: continue
                    avgscore, maxscore, _, avgentro, maxentro, _ = encode_func(example["replaced"], example["replaced_ids"])
                    avgtfidf, maxtfidf, _ = self.get_tfidf_features(example["replaced"], example["replaced_ids"])
                    avgppmi, maxppmi, _ = self.get_ppmi_features(example["replaced"], example["replaced_ids"])
                    features = [avgscore, avgentro, avgtfidf, avgppmi,
                                maxscore, maxentro, maxtfidf, maxppmi]

                    testx.append(features)
                    testy.append(label)
                    cnt += 1
                    if cnt % 500 == 0:
                        print("Test {}".format(cnt))

            fw = codecs.open("feature_combination.txt", "w+", encoding="utf-8")
            infos = []
            for feats in combinations:
                real_trainx = []
                for fs in trainx:
                    new_fs = []
                    for featname in feats:
                        new_fs.append(fs[feature_names[featname]])
                    real_trainx.append(new_fs)
                real_testx = []
                for fs in testx:
                    new_fs = []
                    for featname in feats:
                        new_fs.append(fs[feature_names[featname]])
                    real_testx.append(new_fs)

                self.clf.fit(real_trainx, trainy)
                predy = self.clf.predict(real_testx)

                print("="*20)
                print("Features: {}".format(" ".join(feats)))
                feat_str = "Features: {}".format(" ".join(feats))
                # fw.write("\n\nFeatures: {}\n".format(" ".join(feats)))
                acc, info = binary_eval(predy, testy, return_f1=False)
                infos.append([acc, feat_str, info])
            infos = sorted(infos, key=lambda x:-x[0])

            for item in infos:
                _, feat_str, info = item
                fw.write("\n\n"+feat_str+"\n")
                fw.write(info+"\n")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="svm", type=str) # svm or lr
    parser.add_argument("--device", default="cuda", type=str)

    args = parser.parse_args()

    rep_op = ClfModel(args)

    rep_op.train()

