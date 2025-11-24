import os
import copy
import numpy as np
from tqdm import tqdm

import mindspore as ms
from mindspore import nn, ops, Tensor
from mindspore.ops import functional as F_ms

from util_ms import *
from model_ms import BertForModel, MPNetForModel
from dataloader_ms import build_ms_dataloader  # 假设你前面迁移后的 dataloader


#########################################################
#      动态学习率（warmup + linear decay）
#########################################################
def build_learning_rate(lr, warmup_steps, total_steps):
    """
    完整替代 BertAdam 的 warmup + decay 调度
    """
    warmup_lr = nn.learning_rate_schedule.WarmUpLR(
        learning_rate=lr,
        warmup_steps=warmup_steps
    )

    remain_steps = total_steps - warmup_steps
    decay_lr = nn.learning_rate_schedule.PolynomialDecayLR(
        learning_rate=lr,
        end_learning_rate=0.0,
        decay_steps=remain_steps,
        power=1.0
    )

    return nn.learning_rate_schedule.PiecewiseConstantLR(
        boundaries=[warmup_steps],
        values=[warmup_lr, decay_lr]
    )


#########################################################
#                   PretrainModelManager
#########################################################
class PretrainModelManager:

    def __init__(self, args, data):
        set_seed(args.seed)

        # -------------------------------------------------
        # 选择模型（BERT 或 MPNet）
        # -------------------------------------------------
        if args.bert_model == "sentence-transformers/paraphrase-mpnet-base-v2":
            # ⚠ MPNet 无 MindSpore 权重，这里使用 MPNetForModel 框架 + BERT backbone
            self.model = MPNetForModel(args.bert_model, num_labels=data.n_known_cls)
        else:
            config = BertConfig()
            self.model = BertForModel(config, num_labels=data.n_known_cls)

        # 冻结参数
        if args.freeze_bert_parameters_pretrain:
            self.freeze_parameters(self.model)

        self.data = data

        # -------------------------------------------------
        # 训练步数
        # -------------------------------------------------
        self.total_steps = (
            len(data.train_labeled_examples)
            // args.train_batch_size
        ) * args.num_pretrain_epochs

        warmup_steps = int(self.total_steps * args.warmup_proportion)

        # -------------------------------------------------
        # Optimizer（替代 BertAdam）
        # -------------------------------------------------
        self.learning_rate = build_learning_rate(
            lr=args.lr_pre,
            warmup_steps=warmup_steps,
            total_steps=self.total_steps
        )

        self.optimizer = nn.AdamWeightDecay(
            params=self.model.trainable_params(),
            learning_rate=self.learning_rate,
            weight_decay=1e-2
        )

        self.best_eval_score = 0

    #########################################################
    # 冻结 BERT 参数
    #########################################################
    def freeze_parameters(self, model):
        for name, param in model.bert.parameters_dict().items():
            param.requires_grad = False
            # 只训练高层 + pooler
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True

    #########################################################
    # Forward + Loss for MindSpore training
    #########################################################
    def forward_fn(self, input_ids, input_mask, segment_ids, label_ids):
        loss = self.model(
            input_ids=input_ids,
            token_type_ids=segment_ids,
            attention_mask=input_mask,
            labels=label_ids,
            mode="train"
        )
        return loss

    def grad_fn(self, input_ids, input_mask, segment_ids, label_ids):
        grad_fn = ms.value_and_grad(self.forward_fn, None, self.optimizer.parameters)
        return grad_fn(input_ids, input_mask, segment_ids, label_ids)

    #########################################################
    # Eval
    #########################################################
    def eval(self, args):
        self.model.set_train(False)

        all_logits = []
        all_labels = []

        for batch in self.data.eval_dataloader:
            input_ids, input_mask, segment_ids, label_ids = batch

            logits = self.model(
                input_ids=input_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                mode="eval"
            )[1]

            all_logits.append(logits)
            all_labels.append(label_ids)

        all_logits = ops.concat(all_logits, axis=0)
        all_labels = ops.concat(all_labels, axis=0)

        # softmax
        probs = ops.softmax(all_logits, axis=1)
        preds = ops.argmax(probs, axis=1).asnumpy()
        labels = all_labels.asnumpy()

        from sklearn.metrics import accuracy_score
        acc = round(accuracy_score(labels, preds) * 100, 2)
        return acc

    #########################################################
    # Training
    #########################################################
    def train(self, args):
        print("Start finetune on labeled data")

        wait = 0
        best_model = None

        for epoch in range(args.num_pretrain_epochs):
            self.model.set_train(True)

            epoch_loss = 0
            steps = 0

            # ------------------- Training --------------------
            for batch in self.data.train_labeled_dataloader:
                input_ids, input_mask, segment_ids, label_ids = batch

                loss, grads = self.grad_fn(input_ids, input_mask, segment_ids, label_ids)

                # update
                self.optimizer(grads)
                epoch_loss += loss.asnumpy().item()
                steps += 1

            avg_loss = epoch_loss / steps
            print(f"Epoch {epoch} | loss={avg_loss:.4f}")

            # -------------------- Eval -----------------------
            eval_score = self.eval(args)
            print("Eval score", eval_score)

            if eval_score > self.best_eval_score:
                self.best_eval_score = eval_score
                wait = 0
                best_model = copy.deepcopy(self.model)
            else:
                wait += 1
                if wait >= args.wait_patient:
                    break

        self.model = best_model
        if args.save_model:
            self.save_model(args)

    #########################################################
    # Save pretrained model
    #########################################################
    def save_model(self, args):
        save_dir = args.pretrain_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 保存 checkpoint
        ms.save_checkpoint(self.model, os.path.join(save_dir, "model.ckpt"))
        print(f"Model saved to {save_dir}")
