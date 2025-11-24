import os
import csv
import json
import copy
import random
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

import mindspore as ms
from mindspore import Tensor, nn, ops
from mindspore.dataset import GeneratorDataset

from mindnlp.models import BertModel, BertConfig
from mindnlp.transforms import BertTokenizer

from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import (
    confusion_matrix, normalized_mutual_info_score,
    adjusted_rand_score, accuracy_score
)
from scipy.optimize import linear_sum_assignment


# =====================
#  SET RANDOM SEED
# =====================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    ms.set_seed(seed)


# =====================
#  HUNGRY ALIGNMENT
# =====================
def hungray_aligment(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    return ind, w


def clustering_accuracy_score(y_true, y_pred):
    ind, w = hungray_aligment(y_true, y_pred)
    acc = sum([w[i, j] for i, j in ind]) / y_pred.size
    return acc


def clustering_score(y_true, y_pred):
    return {
        'ACC': round(clustering_accuracy_score(y_true, y_pred)*100, 2),
        'ARI': round(adjusted_rand_score(y_true, y_pred)*100, 2),
        'NMI': round(normalized_mutual_info_score(y_true, y_pred)*100, 2)
    }


# =====================
# Mask tokens (MindSpore)
# =====================
def mask_tokens(inputs: Tensor, tokenizer, special_tokens_mask=None, mlm_probability=0.15):

    labels = ops.identity(inputs)   # clone

    prob_matrix = ops.full(labels.shape, mlm_probability, ms.float32)

    if special_tokens_mask is None:
        mask_list = [
            tokenizer.get_special_tokens_mask(row.asnumpy().tolist(), already_has_special_tokens=True)
            for row in labels
        ]
        special_tokens_mask = Tensor(mask_list, ms.bool_)
    else:
        special_tokens_mask = Tensor(special_tokens_mask, ms.bool_)

    prob_matrix = ops.masked_fill(prob_matrix, special_tokens_mask, 0.0)
    pad_mask = (inputs == 0)
    prob_matrix = ops.masked_fill(prob_matrix, pad_mask, 0.0)

    masked_indices = ops.bernoulli(prob_matrix).astype(ms.bool_)
    labels = ops.masked_fill(labels, ~masked_indices, -100)

    # 80% replace with MASK
    replace_prob = ops.full(labels.shape, 0.8, ms.float32)
    indices_replaced = (ops.bernoulli(replace_prob).astype(ms.bool_) & masked_indices)
    mask_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    inputs = ops.masked_fill(inputs, indices_replaced, mask_id)

    # 10% random token
    random_prob = ops.full(labels.shape, 0.5, ms.float32)
    indices_random = (ops.bernoulli(random_prob).astype(ms.bool_) &
                      masked_indices & ~indices_replaced)

    random_words = ops.randint(0, len(tokenizer), labels.shape, ms.int32)
    inputs = ops.where(indices_random, random_words, inputs)

    return inputs, labels


# =====================
# Data Augmentation Views
# =====================
class view_generator:
    def __init__(self, tokenizer, rtr_prob, seed):
        set_seed(seed)
        self.tokenizer = tokenizer
        self.rtr_prob = rtr_prob

    # 随机 token 替换（MindSpore）
    def random_token_replace(self, ids):
        mask_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        ids, _ = mask_tokens(ids, self.tokenizer, mlm_probability=0.25)
        random_words = ops.randint(0, len(self.tokenizer), ids.shape, ms.int32)

        indices_replaced = (ids == mask_id)
        ids = ops.where(indices_replaced, random_words, ids)
        return ids

    # token shuffle（MindSpore）
    def shuffle_tokens(self, ids):
        view_pos = []
        ids_np = ids.asnumpy()

        for inp in ids_np:
            new_ids = copy.deepcopy(inp)
            special_tokens_mask = self.tokenizer.get_special_tokens_mask(inp, already_has_special_tokens=True)
            sent_tokens_inds = np.where(np.array(special_tokens_mask) == 0)[0]

            shuffled = copy.deepcopy(sent_tokens_inds)
            np.random.shuffle(shuffled)

            inp[sent_tokens_inds] = new_ids[shuffled]
            view_pos.append(inp)

        view_pos = Tensor(np.array(view_pos), ms.int32)
        return view_pos
