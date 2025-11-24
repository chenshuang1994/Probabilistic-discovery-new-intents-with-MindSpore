import os
import copy
import random
import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor
import mindspore.dataset as ds

from util_ms import *
from model_ms import BertForModel, MPNetForModel
from dataloader_ms import build_ms_dataset
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from mindspore import context
from init_parameter import init_model
from dataloader import Data
from pretrain_ms import PretrainModelManagerMS   # 你在 Part3 中已迁移
from model_manager_ms import ModelManagerMS 

#########################################################
# 固定随机种子
#########################################################
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    ms.set_seed(seed)


#########################################################
# ModelManager — Part 1
#########################################################
class ModelManager:
    def __init__(self, args, data, pretrained_model=None):
        set_seed(args.seed)

        ########################################
        # 1. 加载预训练模型（BERT or MPNet）
        ########################################
        if pretrained_model is None:
            # MindSpore 不支持 .from_pretrained("./model") 的 PyTorch 权重
            # 因此我们加载 MindNLP 的 BERT config
            config = BertConfig()
            pretrained_model = BertForModel(config, num_labels=data.n_known_cls)

        # load MTP checkpoint
        if args.load_mtp:
            # ⚠ MindSpore 无法直接 load PyTorch .bin，这部分需要你提供 MS ckpt 才能加载
            print("[Warning] MindSpore cannot load PyTorch .bin weights. Skipping load_mtp.")

        ########################################
        # 2. 冻结参数
        ########################################
        if args.freeze_bert_parameters_em:
            self.freeze_parameters(pretrained_model)

        self.model = pretrained_model  # MindSpore Cell

        ########################################
        # 3. 确定聚类类别数 K
        ########################################
        if args.cluster_num_factor > 1:
            self.num_labels = self.predict_k(args, data)
        else:
            self.num_labels = data.num_labels

        ########################################
        # 4. 训练步数
        ########################################
        num_train_examples = (
            len(data.train_labeled_examples)
            + len(data.train_unlabeled_examples)
        )
        self.num_train_optimization_steps = (
            num_train_examples // args.train_batch_size
        ) * args.num_train_epochs

        ########################################
        # 5. Optimizer (替代 BertAdam)
        ########################################
        warmup_steps = int(self.num_train_optimization_steps * args.warmup_proportion)
        
        # warmup + linear decay
        lr_schedule = nn.learning_rate_schedule.PiecewiseConstantLR(
            boundaries=[warmup_steps],
            values=[
                nn.learning_rate_schedule.WarmUpLR(args.lr, warmup_steps),
                nn.learning_rate_schedule.PolynomialDecayLR(
                    learning_rate=args.lr,
                    end_learning_rate=0.0,
                    decay_steps=self.num_train_optimization_steps - warmup_steps,
                    power=1.0
                )
            ]
        )

        self.optimizer = nn.AdamWeightDecay(
            params=self.model.trainable_params(),
            learning_rate=lr_schedule,
            weight_decay=0.01
        )

        ########################################
        # 6. Buffers
        ########################################
        self.centroids = None
        self.best_eval_score = 0
        self.test_results = None
        self.predictions = None
        self.true_labels = None

        self.data = data
#########################################################
# Part 2: alignment() + get_features_labels()
#########################################################

from mindspore import ops, Tensor


class ModelManager(ModelManager):  # 继承上一部分
    #########################################################
    # 1. 提取所有特征 & labels
    #########################################################
    def get_features_labels(self, dataloader, model, args):
        """
        MindSpore 的 dataset 返回的是 Python dict，Tensor 已经在 CPU/GPU 中
        """
        model.set_train(False)

        total_features = []
        total_labels = []

        for batch in dataloader.create_dict_iterator():
            input_ids = batch["input_ids"]
            segment_ids = batch["token_type_ids"]
            input_mask = batch["attention_mask"]
            label_ids = batch["labels"]

            # 得到 pooled feature
            feature = model(
                input_ids=input_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                feature_ext=True
            )
            total_features.append(feature)
            total_labels.append(label_ids)

        # concat
        total_features = ops.concat(total_features, axis=0)
        total_labels = ops.concat(total_labels, axis=0)

        return total_features, total_labels

    #########################################################
    # 2. 质心对齐（匈牙利算法）
    #########################################################
    def alignment(self, km, args):
        """
        km: sklearn KMeans 结果
        alignment: 处理聚类质心变化导致的 label mismatch
        """
        if self.centroids is not None:
            old_centroids = self.centroids.asnumpy()   # (K, D)
            new_centroids = km.cluster_centers_        # (K, D)

            # 欧式距离矩阵 (K x K)
            distance_matrix = np.linalg.norm(
                old_centroids[:, None, :] - new_centroids[None, :, :], axis=2
            )

            # 匈牙利算法
            row_ind, col_ind = linear_sum_assignment(distance_matrix)

            # 根据对齐结果重排新质心
            new_centroids = Tensor(new_centroids[col_ind], ms.float32)
            self.centroids = new_centroids

            # 伪标签映射 old → new
            mapping = {old: new for new, old in enumerate(col_ind)}
            pseudo_labels = np.array([mapping[label] for label in km.labels_])

        else:
            # 第一次训练不需要对齐
            self.centroids = Tensor(km.cluster_centers_, ms.float32)
            pseudo_labels = km.labels_

        pseudo_labels = Tensor(pseudo_labels, ms.int32)
        return pseudo_labels

#########################################################
# Part 3.1: 更新伪标签 → Dataset
#########################################################
def update_pseudo_labels(self, pseudo_labels, args,
                         input_ids, input_mask, segment_ids, label_ids):
    """
    pseudo_labels: (N,) Tensor
    其余 input_xxx: (N, L) Tensor
    """

    dataset_dict = {
        "input_ids": input_ids.asnumpy(),
        "attention_mask": input_mask.asnumpy(),
        "token_type_ids": segment_ids.asnumpy(),
        "pseudo_labels": pseudo_labels.asnumpy(),
        "labels": label_ids.asnumpy()
    }

    train_dataset = ds.NumpySlicesDataset(dataset_dict, shuffle=True)
    train_dataset = train_dataset.batch(args.train_batch_size)

    return train_dataset
import mindspore as ms
from mindspore import ops, nn, Tensor

###############################################
# Part 4: MindSpore 训练循环
###############################################

def train(self, args, data):
    print("==== MindSpore Training Start ====")

    bestresults = {'ACC': 0, 'ARI': 0, 'NMI': 0}
    jsonresults = {}

    # 创建交叉熵损失
    ce_loss_fn = nn.CrossEntropyLoss()

    # 定义反向传播函数：计算 loss 与 grads
    def forward_fn(input_ids, token_type_ids, attention_mask,
                   pseudo_label_ids, label_ids):
        """
        input_xxx: Tensor(batch, L)
        """
        # ================================
        # 1. 对比学习损失（CL Loss）
        # ================================

        # 标签矩阵 (batch, batch)
        batch_size = input_ids.shape[0]
        pseudo = pseudo_label_ids
        label_matrix = ops.zeros((batch_size, batch_size), ms.float32)

        for i in range(batch_size):
            label_matrix[i] = (pseudo == pseudo[i]).astype(ms.float32)

        # 模型得到特征
        feats = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            mode="sim",
            feature_ext=True
        )   # (B, D)

        feats = ops.L2Normalize()(feats)

        # 相似度矩阵
        sim_matrix = ops.exp(ops.matmul(feats, feats.T) / args.t)
        sim_matrix = sim_matrix - ops.diag(sim_matrix.diagonal())

        # 正样本矩阵
        pos_matrix = ops.zeros_like(sim_matrix)
        pos_mask_np = (label_matrix.asnumpy() != 0)
        pos_matrix = pos_matrix.asnumpy()
        sim_matrix_np = sim_matrix.asnumpy()
        pos_matrix[pos_mask_np] = sim_matrix_np[pos_mask_np]
        pos_matrix = Tensor(pos_matrix, ms.float32)

        # CL loss = -log (positive / sum_row)
        denom = sim_matrix.sum(axis=1).reshape((-1, 1))
        cl_ratio = pos_matrix / denom
        cl_vals = cl_ratio[cl_ratio > 0]
        cl_loss = -ops.log(cl_vals).mean()

        if cl_loss.isnan():
            cl_loss = Tensor(0.0, ms.float32)

        # ================================
        # 2. CE loss（仅有标签的部分）
        # ================================
        if (label_ids >= 0).any():
            mask = (label_ids >= 0)
            ce_input_ids = input_ids[mask]
            ce_token_type_ids = token_type_ids[mask]
            ce_attention_mask = attention_mask[mask]
            ce_label_ids = label_ids[mask]

            _, logits = self.model(
                input_ids=ce_input_ids,
                token_type_ids=ce_token_type_ids,
                attention_mask=ce_attention_mask,
                labels=None,
                mode="train"
            )
            ce_loss = ce_loss_fn(logits, ce_label_ids)
        else:
            ce_loss = Tensor(0.0, ms.float32)

        # ================================
        # 总损失
        # ================================
        loss = (1 - args.beta) * cl_loss + args.beta * ce_loss
        return loss

    # 绑定正向与反向图
    grad_fn = ops.value_and_grad(forward_fn, None,
                                 weights=self.optimizer.parameters)

    # ===========================================
    # 外层 epoch 循环
    # ===========================================
    for epoch in range(args.num_train_epochs):
        print(f"\n===== Epoch {epoch} =====")

        # step1：提取特征并聚类
        feats, labels = self.get_features_labels(data.train_semi_dataloader,
                                                 self.model,
                                                 args)
        feats_np = feats.asnumpy()

        km = KMeans(n_clusters=self.num_labels).fit(feats_np)
        pseudo_labels = self.alignment(km, args)  # Tensor(N)

        # 若使用增强数据
        if args.augment_data:
            updated_semi_label = self.update_dataset(km, feats_np, args.k)
            train_dataset = self.update_pseudo_labels(
                pseudo_labels, args,
                data.semi_input_ids,
                data.semi_input_mask,
                data.semi_segment_ids,
                Tensor(updated_semi_label, ms.int32)
            )
        else:
            train_dataset = self.update_pseudo_labels(
                pseudo_labels, args,
                data.semi_input_ids,
                data.semi_input_mask,
                data.semi_segment_ids,
                data.semi_label_ids
            )

        # ================================
        # step2：训练一个 epoch
        # ================================
        running_loss = 0
        steps = 0

        for batch in train_dataset.create_dict_iterator():
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            token_type_ids = batch["token_type_ids"]
            pseudo_label_ids = batch["pseudo_labels"]
            label_ids = batch["labels"]

            (loss), grads = grad_fn(
                input_ids, token_type_ids, attention_mask,
                pseudo_label_ids, label_ids
            )

            self.optimizer(grads)
            running_loss += loss.asnumpy()
            steps += 1

        epoch_loss = running_loss / steps
        print(f"Epoch {epoch} loss = {epoch_loss:.4f}")

        # ================================
        # step3：评估
        # ================================
        feats, labels = self.get_features_labels(data.test_dataloader,
                                                 self.model,
                                                 args)
        feats_np = feats.asnumpy()
        km = KMeans(n_clusters=self.num_labels).fit(feats_np)

        y_pred = km.labels_
        y_true = labels.asnumpy()

        results = clustering_score(y_true, y_pred)
        jsonresults[epoch] = results

        print("Evaluation:", results)

        # update best
        if sum(results.values()) > sum(bestresults.values()):
            bestresults = results

        # 写入 json
        with open(f'./outputs/info_{args.name}.json', 'w') as f:
            import json
            f.write(json.dumps({"best": bestresults,
                                "history": jsonresults}, indent=4))

    print("===== Training Finished =====")
#########################################################
# Part 5.1  MindSpore evaluation()
#########################################################
def evaluation(self, args, data):
    # 提取特征
    feats, labels = self.get_features_labels(data.test_dataloader,
                                             self.model,
                                             args)
    feats_np = feats.asnumpy()
    labels_np = labels.asnumpy()

    # 聚类
    km = KMeans(n_clusters=self.num_labels).fit(feats_np)
    y_pred = km.labels_
    y_true = labels_np

    # 计算 ACC / ARI / NMI
    results = clustering_score(y_true, y_pred)
    print("results", results)

    # Hungarian 对齐
    ind, _ = hungray_aligment(y_true, y_pred)
    mapping = {i[0]: i[1] for i in ind}
    aligned_pred = np.array([mapping[idx] for idx in y_pred])

    # 混淆矩阵
    cm = confusion_matrix(y_true, aligned_pred)
    print("confusion matrix\n", cm)

    self.test_results = results
    self.save_results(args)
#########################################################
# Part 5.2 MindSpore eval_pretrain()
#########################################################
def eval_pretrain(self):
    self.model.set_train(False)

    total_labels = []
    total_logits = []

    for batch in self.data.eval_dataloader.create_dict_iterator():
        input_ids = batch["input_ids"]
        token_type_ids = batch["token_type_ids"]
        attention_mask = batch["attention_mask"]
        label_ids = batch["labels"]

        _, logits = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            mode="eval"
        )

        total_labels.append(label_ids)
        total_logits.append(logits)

    total_labels = ops.concat(total_labels, axis=0)
    total_logits = ops.concat(total_logits, axis=0)

    probs = ops.softmax(total_logits, axis=1)
    preds = ops.Argmax(axis=1)(probs)

    y_pred = preds.asnumpy()
    y_true = total_labels.asnumpy()
    acc = round(accuracy_score(y_true, y_pred) * 100, 2)
    return acc
#########################################################
# Part 5.3 MindSpore save_results()
#########################################################
def save_results(self, args):
    if not os.path.exists(args.save_results_path):
        os.makedirs(args.save_results_path)

    names = ['dataset', 'alpha', 'beta', 'batch_size',
             'seed', 'K', 'name']
    var = [
        args.dataset,
        args.t,
        args.beta,
        args.train_batch_size,
        args.seed,
        self.num_labels,
        args.name
    ]
    vars_dict = {k: v for k, v in zip(names, var)}

    results = dict(self.test_results, **vars_dict)
    keys = list(results.keys())
    values = list(results.values())

    file_name = f"{args.dataset}.csv"
    results_path = os.path.join(args.save_results_path, file_name)

    if not os.path.exists(results_path):
        df = pd.DataFrame([values], columns=keys)
        df.to_csv(results_path, index=False)
    else:
        df = pd.read_csv(results_path)
        df.loc[len(df)] = values
        df.to_csv(results_path, index=False)

    print("test_results:")
    print(pd.read_csv(results_path))
############################################################
# Part 6: MindSpore 版本 run_ms.py 主入口
############################################################

if __name__ == '__main__':

    # 设置运行环境（GPU 模式）
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    parser = init_model()
    args = parser.parse_args()

    # 如果你希望指定 GPU：
    # context.set_context(device_id=int(args.gpu_id))

    print("==========  Preparing data  ==========")
    data = Data(args)

    # ================================
    # 选择是否进行 pretrain
    # ================================
    if args.pretrain:
        print("==========  Pretraining  ==========")
        manager_p = PretrainModelManagerMS(args, data)
        manager_p.train(args)

        # 预训练后的模型作为主模型初始化输入
        manager = ModelManagerMS(args, data, pretrained_model=manager_p.model)

    else:
        # 不进行预训练，直接初始化空模型
        args.pretrain_dir = 'pretrained_' + args.dataset
        manager = ModelManagerMS(args, data)

    # ================================
    # 主训练
    # ================================
    print("==========  Training begin  ==========")
    manager.train(args, data)
    print("==========  Training finished  ==========")

    # ================================
    # 最终测试 evaluation
    # ================================
    print("==========  Evaluation begin  ==========")
    manager.evaluation(args, data)
    print("==========  Evaluation finished  ==========")
