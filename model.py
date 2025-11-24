import mindspore as ms
from mindspore import nn, ops, Tensor
import numpy as np

from mindnlp.models import BertModel, BertConfig
from mindnlp.transforms import BertTokenizer


###########################################
# Mean Pooling (MindSpore)
###########################################
def mean_pooling(token_embeddings, attention_mask):
    """
    token_embeddings: [batch, seq, hidden]
    attention_mask: [batch, seq]
    """
    mask_expanded = ops.expand_dims(attention_mask, -1)
    mask_expanded = ops.broadcast_to(mask_expanded, token_embeddings.shape)
    mask_expanded = mask_expanded.astype(ms.float32)

    sum_embeddings = ops.sum(token_embeddings * mask_expanded, axis=1)
    sum_mask = ops.clip_by_value(ops.sum(mask_expanded, axis=1), 1e-9, 1e9)

    return sum_embeddings / sum_mask


###########################################
# MindSpore BERT Model for Our Task
###########################################
class BertForModel(nn.Cell):
    def __init__(self, config: BertConfig, num_labels: int):
        super(BertForModel, self).__init__()
        self.num_labels = num_labels

        # backbone
        self.bert = BertModel(config)

        hidden = config.hidden_size
        drop = config.hidden_dropout_prob

        # dense
        self.dense = nn.SequentialCell(
            nn.Dense(hidden, hidden),
            nn.Tanh(),
            nn.Dropout(1 - drop)
        )

        # projection (for contrastive learning)
        self.proj = nn.SequentialCell(
            nn.Dense(hidden, 2 * hidden),
            nn.ReLU(),
            nn.Dropout(1 - drop),
            nn.Dense(2 * hidden, hidden),
            nn.ReLU(),
            nn.Dropout(1 - drop)
        )

        # classifier
        self.classifier = nn.Dense(hidden, num_labels)

        self.ce_loss = nn.CrossEntropyLoss()

    def construct(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        mode=None,
        centroids=None,
        labeled=False,
        feature_ext=False
    ):
        """
        mode:
            - "sim": return proj_output
            - feature_ext=True: return pooled_output
            - "train": return CE loss
            - else: return (pooled_output, logits)
        """

        # encoded: (sequence_output, pooled_output)
        seq_output, pooled_output_ = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )

        # we use mean pooling instead of pooler
        pooled_output = mean_pooling(seq_output, attention_mask)

        pooled_output = self.dense(pooled_output)
        proj_output = self.proj(pooled_output)
        logits = self.classifier(pooled_output)

        if mode == "sim":
            return proj_output

        if feature_ext:
            return pooled_output

        if mode == "train":
            return self.ce_loss(logits, labels)

        return pooled_output, logits


###########################################
# MindSpore MPNet Model (structure only)
# NOTE: MindSpore does not provide MPNet weights
#       You can load another backbone as replacement.
###########################################
class MPNetForModel(nn.Cell):
    def __init__(self, model_name: str, num_labels: int):
        super(MPNetForModel, self).__init__()

        # ⚠ MPNet 不存在 MindSpore 官方实现
        # 你可以选择改成 BERT、RoBERTa 或自定义 encoder
        config = BertConfig()  # 使用 BERT config 代替
        self.backbone = BertModel(config)

        hidden = config.hidden_size
        drop = config.hidden_dropout_prob

        self.dense = nn.SequentialCell(
            nn.Dense(hidden, hidden),
            nn.Tanh(),
            nn.Dropout(1 - drop)
        )

        self.proj = nn.SequentialCell(
            nn.Dense(hidden, 2*hidden),
            nn.ReLU(),
            nn.Dropout(1 - drop),
            nn.Dense(2*hidden, hidden),
            nn.ReLU(),
            nn.Dropout(1 - drop)
        )

        self.classifier = nn.Dense(hidden, num_labels)
        self.ce_loss = nn.CrossEntropyLoss()

    def construct(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        mode=None,
        feature_ext=False
    ):
        seq_output, pooled_output_ = self.backbone(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )

        pooled_output = mean_pooling(seq_output, attention_mask)
        pooled_output = self.dense(pooled_output)

        proj_output = self.proj(pooled_output)
        logits = self.classifier(pooled_output)

        if mode == "sim":
            return proj_output
        if feature_ext:
            return pooled_output
        if mode == "train":
            return self.ce_loss(logits, labels)

        return pooled_output, logits
