# coding=utf-8
import os
import sys
import csv
import tqdm
import copy
import random
import numpy as np

import mindspore as ms
from mindspore import Tensor, ops

from mindnlp.transforms import BertTokenizer
from mindnlp.models import BertModel

from scipy.spatial.distance import cosine
from mindspore.dataset import GeneratorDataset


########################################
# 固定随机种子
########################################
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    ms.set_seed(seed)


########################################
# 工具函数
########################################
def difference(a, b):
    _b = set(b)
    return [item for item in a if item not in _b]


########################################
# InputExample / InputFeatures
########################################
class InputExample:
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures:
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


########################################
# TSV 数据读取
########################################
class DataProcessor:
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = [line for line in reader]
            return lines


########################################
# DatasetProcessor
########################################
class DatasetProcessor(DataProcessor):
    def get_examples(self, data_dir, mode):
        if mode == 'train':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
        elif mode == 'eval':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
        elif mode == 'test':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self, data_dir):
        import pandas as pd
        df = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep="\t")
        return list(np.unique(np.array(df['label'])))

    def _create_examples(self, lines, set_type):
        examples = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            if len(line) != 2:
                continue
            guid = f"{set_type}-{i}"
            text_a = line[0]
            label = line[1]
            examples.append(InputExample(guid, text_a, None, label))
        return examples


########################################
# feature 转换
########################################
def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_len = len(tokens_a) + len(tokens_b)
        if total_len <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)
        else:
            tokens_b.pop()


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for example in examples:
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        label_id = label_map[example.label]

        features.append(InputFeatures(
            input_ids, input_mask, segment_ids, label_id
        ))
    return features


########################################
# MindSpore Dataloader（替代 PyTorch DataLoader）
########################################
def ms_dataset_from_tensors(tensors, batch_size, shuffle=False):
    def generator():
        for i in range(len(tensors[0])):
            yield tuple(t[i] for t in tensors)

    dataset = GeneratorDataset(
        generator, column_names=["input_ids", "input_mask", "segment_ids", "label_ids"],
        shuffle=shuffle
    )
    dataset = dataset.batch(batch_size)
    return dataset


########################################
# Data 类（核心）
########################################
class Data:
    def __init__(self, args):
        set_seed(args.seed)

        max_seq_lengths = {"clinc": 30, "stackoverflow": 45, "banking": 55}
        args.max_seq_length = max_seq_lengths[args.dataset]

        processor = DatasetProcessor()
        self.data_dir = os.path.join(args.data_dir, args.dataset)
        self.all_label_list = list(map(str, processor.get_labels(self.data_dir)))

        self.n_known_cls = round(len(self.all_label_list) * args.known_cls_ratio)
        self.known_label_list = list(np.random.choice(self.all_label_list, self.n_known_cls, replace=False))
        self.unknown_label_list = difference(self.all_label_list, self.known_label_list)

        self.num_labels = int(len(self.all_label_list) * args.cluster_num_factor)

        # prepare datasets
        self.train_labeled_examples, self.train_unlabeled_examples = \
            self.get_examples(processor, args, "train")
        self.eval_examples = self.get_examples(processor, args, "eval")
        self.test_examples = self.get_examples(processor, args, "test")

        # dataloaders
        self.train_labeled_dataloader = self.get_loader(self.train_labeled_examples, args)
        self.eval_dataloader = self.get_loader(self.eval_examples, args, mode="eval")
        self.test_dataloader = self.get_loader(self.test_examples, args, mode="test")

    ########################################
    # 获取训练/测试数据
    ########################################
    def get_examples(self, processor, args, mode, separate=False):
        ori_examples = processor.get_examples(self.data_dir, mode)

        if mode == "train":
            labels = np.array([e.label for e in ori_examples])
            labeled_ids = []

            for label in self.known_label_list:
                pos = np.where(labels == label)[0]
                num = round(len(pos) * args.labeled_ratio)
                labeled_ids.extend(random.sample(pos.tolist(), num))

            labeled = [ori_examples[i] for i in labeled_ids]
            unlabeled = [e for i, e in enumerate(ori_examples) if i not in labeled_ids]
            return labeled, unlabeled

        elif mode == "eval":
            return [e for e in ori_examples if e.label in self.known_label_list]

        elif mode == "test":
            return ori_examples

    ########################################
    # semi 数据
    ########################################
    def get_loader(self, examples, args, mode="train"):
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)

        if mode == "train":
            features = convert_examples_to_features(
                examples, self.known_label_list, args.max_seq_length, tokenizer)
        else:
            features = convert_examples_to_features(
                examples, self.all_label_list, args.max_seq_length, tokenizer)

        input_ids = Tensor([f.input_ids for f in features], ms.int32)
        input_mask = Tensor([f.input_mask for f in features], ms.int32)
        segment_ids = Tensor([f.segment_ids for f in features], ms.int32)
        label_ids = Tensor([f.label_id for f in features], ms.int32)

        dataset = ms_dataset_from_tensors(
            (input_ids, input_mask, segment_ids, label_ids),
            batch_size=args.train_batch_size if mode == "train" else args.eval_batch_size,
            shuffle=True if mode == "train" else False
        )
        return dataset
