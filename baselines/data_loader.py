from torch.utils import data
import torch
import numpy as np
from utils import remove_marked_sen
from tqdm import tqdm, trange
import collections
import codecs
import json


class InputExample(object):

    def __init__(self, guid, sen, idxs, label):
        self.guid = guid
        self.sen = sen
        self.idxs = idxs
        self.label = label


class InputFeatures(object):

    def __init__(self, guid, input_ids, input_mask, segment_ids,  predict_mask, label_id):
        self.guid = guid
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.predict_mask = predict_mask
        self.label_id = label_id


class DataProcessor(object):
    def __init__(self):
        self.num_consist = 0
        self.num_hallu = 0

    def _read_data(self, input_file, require_uidx=False):
        with open(input_file) as f:
            # out_lines = []
            out_lists = []
            entries = f.read().strip().split("\n")
            for entry in entries:
                example = json.loads(entry.strip())
                if "hallucination" not in example:
                    label = -1
                else:
                    label = example["hallucination"]
                    if label not in [0, 1]:
                        continue
                if require_uidx:
                    sen, token_ids, uidx = example["replaced"], example["replaced_ids"], example["idx"]
                    out_lists.append([sen, token_ids, label, uidx])
                else:
                    sen, token_ids = example["replaced"], example["replaced_ids"]
                    out_lists.append([sen, token_ids, label])
        return out_lists

    def _create_examples(self, all_lists):
        examples = []
        for (i, one_lists) in enumerate(all_lists):
            guid = i
            if len(one_lists) == 3:  # Don't contain key "idx" in json file
                sen, token_ids, label = one_lists
            elif len(one_lists) == 4:  # Contain key "idx" in json file
                sen, token_ids, label, guid = one_lists
            else:
                assert len(one_lists) == 3 or len(one_lists) == 4

            if label == 0:
                self.num_consist += 1
            elif label == 1:
                self.num_hallu += 1

            examples.append(InputExample(
                guid=guid, sen=sen, idxs=token_ids, label=label))
        return examples

    def get_examples(self, path, require_uidx=False):
        return self._create_examples(
            self._read_data(path, require_uidx))

    def get_label_dist(self):
        return [self.num_consist, self.num_hallu]

def truncate(rep_subtokens, predict_mask, max_seq_length, rep_start_id, rep_end_id, mode="offline"):
    '''
    Truncate the sequence if given a fixed context window. For example, given the following input sentence:
    "he signed a professional contract and promoted to the ===senior team=== where he managed to play for almost 3 years ."
    if the context window length is set as 4, the function will truncate the input as follows:

    online mode: "and promoted to the ===senior team==="
    offline mode: "to the ===senior team=== where he"
    '''
    if mode == "offline":
        if len(rep_subtokens) > max_seq_length - 2:
            mid_pt = int((rep_start_id + rep_end_id) / 2)
            left_seq_length = int(max_seq_length / 2)
            right_seq_length = max_seq_length - left_seq_length
            if mid_pt - left_seq_length >= 0 and mid_pt + right_seq_length < len(rep_subtokens):
                left_pt = mid_pt - left_seq_length
                right_pt = mid_pt + right_seq_length
            elif mid_pt - left_seq_length < 0 and mid_pt + right_seq_length < len(rep_subtokens):
                left_pt = 0
                right_pt = max_seq_length
            elif mid_pt - left_seq_length >= 0 and mid_pt + right_seq_length >= len(rep_subtokens):
                right_pt = len(rep_subtokens)
                left_pt = len(rep_subtokens) - max_seq_length
            elif mid_pt - left_seq_length < 0 and mid_pt + right_seq_length >= len(rep_subtokens):
                left_pt = 0
                right_pt = len(rep_subtokens)
            rep_subtokens = rep_subtokens[left_pt:right_pt - 1]
            predict_mask = predict_mask[left_pt:right_pt - 1]
    else: # online
        left_pt, right_pt = 0, rep_end_id + 1
        if right_pt > max_seq_length - 2:
            left_pt = right_pt - (max_seq_length - 2)
        rep_subtokens = rep_subtokens[left_pt:right_pt]
        predict_mask = predict_mask[left_pt:right_pt]
    return rep_subtokens, predict_mask


def example2feature(example, tokenizer, max_seq_length, model_name, mode="offline"):
    rep_start_id, rep_end_id = example.idxs
    rep_tokens = remove_marked_sen(example.sen, rep_start_id, rep_end_id)

    if 'xlnet' in model_name.lower():
        rep_subtokens = []
        predict_mask = []

        for id, rep_token in enumerate(rep_tokens):
            rep_subtoken = tokenizer.tokenize(rep_token)
            if id >= rep_start_id and id <= rep_end_id:
                rep_subtokens.extend(rep_subtoken)
                predict_mask.extend(len(rep_subtoken) * [1])
            else:
                rep_subtokens.extend(rep_subtoken)
                predict_mask.extend(len(rep_subtoken) * [0])

        rep_subtokens, predict_mask = truncate(rep_subtokens, predict_mask, max_seq_length, rep_start_id, rep_end_id, mode=mode)

        rep_subtokens.extend(["<sep>", "<cls>"])
        predict_mask.extend([0, 0])

    elif 'gpt' not in model_name.lower():

        rep_subtokens = []
        predict_mask = []
        for id, rep_token in enumerate(rep_tokens):
            rep_subtoken = tokenizer.tokenize(rep_token)
            if id >= rep_start_id and id <= rep_end_id:
                rep_subtokens.extend(rep_subtoken)
                predict_mask.extend(len(rep_subtoken) * [1])
            else:
                rep_subtokens.extend(rep_subtoken)
                predict_mask.extend(len(rep_subtoken) * [0])

        rep_subtokens, predict_mask = truncate(rep_subtokens, predict_mask, max_seq_length, rep_start_id, rep_end_id, mode=mode)

        rep_subtokens.insert(0, "[CLS]")
        predict_mask.insert(0, 0)
        rep_subtokens.append('[SEP]')
        predict_mask.append(0)

    elif 'gpt' in model_name.lower():
        rep_subtokens = []
        predict_mask = []

        for id, rep_token in enumerate(rep_tokens):
            rep_token = " "+rep_token if id!=0 else rep_token
            rep_subtoken = tokenizer.tokenize(rep_token)
            if id >= rep_start_id and id <= rep_end_id:
                rep_subtokens.extend(rep_subtoken)
                predict_mask.extend(len(rep_subtoken) * [1])
            else:
                rep_subtokens.extend(rep_subtoken)
                predict_mask.extend(len(rep_subtoken) * [0])

        rep_subtokens, predict_mask = truncate(rep_subtokens, predict_mask, max_seq_length, rep_start_id, rep_end_id, mode=mode)

    input_ids = tokenizer.convert_tokens_to_ids(rep_subtokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)

    feat=InputFeatures(
                guid=example.guid,
                # tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                predict_mask=predict_mask,
                label_id=example.label)
    return feat

def get_examples_from_sen_tuple(sen, rep_pos):
    examples = []
    for uid, pos in enumerate(rep_pos):
        examples.append(InputExample(guid=uid, sen=sen, idxs=pos, label=0))
    return examples

class HalluDataset(data.Dataset):
    def __init__(self, examples, tokenizer, max_seq_length, model_name, task_mode="offline"):
        self.examples=examples
        self.tokenizer=tokenizer
        self.max_seq_length=max_seq_length
        self.model_name = model_name
        self.task_mode = task_mode

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        feat=example2feature(self.examples[idx], self.tokenizer, self.max_seq_length,
                             self.model_name, self.task_mode)
        return feat.input_ids, feat.input_mask, feat.segment_ids, feat.predict_mask, feat.label_id, feat.guid

    @classmethod
    def pad(cls, batch):

        seqlen_list = [len(sample[0]) for sample in batch]
        maxlen = np.array(seqlen_list).max()

        f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: X for padding
        input_ids_list = torch.LongTensor(f(0, maxlen))
        input_mask_list = torch.LongTensor(f(1, maxlen))
        segment_ids_list = torch.LongTensor(f(2, maxlen))
        predict_mask_list = torch.ByteTensor(f(3, maxlen))
        label_id = torch.LongTensor([sample[4] for sample in batch])
        guids = [sample[5] for sample in batch]

        return input_ids_list, input_mask_list, segment_ids_list, predict_mask_list, label_id, guids

'''
trainpath="../data_collections/wiki5k+sequential+topk_10+temp_1+context_1+rep_0.6+sample_1.annotate.finish"
# trainpath="../data_collections/wiki5k+sequential+topk_10+temp_1+context_1+rep_0.6+sample_1.annotate"
dp = DataProcessor()
train_examples = dp.get_examples(trainpath)
print(len(train_examples))
print(train_examples[2].sen)
print(train_examples[2].idxs)
print(train_examples[2].guid)
print(dp.get_label_dist())

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_dataset = HalluDataset(train_examples,tokenizer, 512, "bert")

train_dataloader = data.DataLoader(dataset=train_dataset,
                                   batch_size=32,
                                   shuffle=False,
                                   num_workers=4,
                                   collate_fn=HalluDataset.pad)

for step, batch in enumerate(train_dataloader):
    input_ids, input_mask, segment_ids, predict_mask, label_ids, guids = batch
    print("id {} mask {} segment {}, pmask {}, label {}".format(input_ids.size(), input_mask.size(),
                                                                segment_ids.size(), predict_mask.size(), label_ids.size()))
    print("guid {}".format(" ".join([str(guid) for guid in guids])))
'''