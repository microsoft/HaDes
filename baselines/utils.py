from torch import nn
import torch
from torch.nn import functional as F
import codecs
import json
import spacy
from sklearn.metrics import classification_report, accuracy_score, hamming_loss, \
    f1_score, precision_score, recall_score, average_precision_score, roc_auc_score, confusion_matrix, \
    brier_score_loss
import numpy as np


def binary_eval(predy, testy, verbose=True, return_f1=False, predscore=None):
    acc = accuracy_score(testy, predy)
    f1 = f1_score(testy, predy, average=None)
    precision = precision_score(testy, predy, average=None)
    recall = recall_score(testy, predy, average=None)
    epsilon = 1e-8

    htn, hfp, hfn, htp = confusion_matrix(testy, predy).ravel()
    hsensi = htp / (htp + hfn + epsilon)
    hspec = htn / (hfp + htn + epsilon)
    gmean = np.sqrt(hsensi*hspec)


    info = "Acc : {}\nf1 : {}\nprecision : {}\nrecall : {}\nG-mean : {}".format(acc,
            " ".join([str(x) for x in f1]), " ".join([str(x) for x in precision]),
            " ".join([str(x) for x in recall]), gmean)

    if predscore is not None:
        bss = brier_score_loss(testy, predscore)
        roc_auc = roc_auc_score(testy, predscore)
        info += "\nbss : {}\nROC-AUC : {}".format(bss, roc_auc)

    if verbose:
        print(info)

    if return_f1:
        return acc, f1, precision, recall, gmean, bss, roc_auc, info
    else:
        return acc, info


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


def sent_ner_bounds(sen, nlp=None):
    if nlp is None:
        nlp = spacy.load('en')
    tokens, tags = [], []
    print(sen)
    for doc in nlp.pipe([sen]):
        for token in doc:
            tags.append(token.ent_iob_)
            tokens.append(str(token))

    rep_pos = []
    vis = [False for _ in range(len(tags))]
    for idx, tag in enumerate(tags):
        if tag == 'O':
            rep_pos.append([idx, idx])
            vis[idx] = True
        elif tag == 'B':
            end = idx
            for j in range(idx+1, len(tags)):
                if tags[j] == 'I':
                    end = j
                else:
                    break
            rep_pos.append([idx, end])
        elif tag == 'I':
            continue

    return ' '.join(tokens), rep_pos


def remove_marked_sen(sen, start_id, end_id):
    tokens = sen if type(sen) == list else sen.strip().split()
    if tokens[start_id].startswith("===") and tokens[end_id].endswith("==="):
        tokens[start_id] = tokens[start_id][3:]
        tokens[end_id] = tokens[end_id][:-3]
    return tokens

