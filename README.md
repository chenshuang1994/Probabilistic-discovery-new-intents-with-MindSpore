# A Probabilistic Framework for Discovering New Intents with **MindSpore**

## 🔍 Introduction

本仓库提供论文 **“A Probabilistic Framework for Discovering New Intents”** 的 **MindSpore 官方实现版本**。
与原始的 PyTorch 实现不同，本项目基于 MindSpore 对模型结构、训练流程、MTP（Multi-Task Pretraining）相关模块进行了适配与重构，以更好地支持 Ascend/GPU 环境。

---

## 📦 Dependencies

### 1. 创建 Conda 环境（建议 Python 3.9）

```bash
conda create -n nid_ms python=3.9
conda activate nid_ms
```

### 2. 安装 MindSpore

**Ascend**：

```bash
pip install mindspore -f https://www.mindspore.cn/whl/ascend910/
```

**GPU**：

```bash
pip install mindspore-gpu
```

### 3. 安装所需第三方依赖

```bash
pip install -r requirements.txt
```

---

## 🧩 Model Preparation

### 1. 获取 BERT 预训练模型

下载原始 **BERT-base-uncased** 模型（TensorFlow 格式）：

[https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip)

### 2. 转换为 MindSpore 权重

使用 MindSpore 官方转换工具（或自带脚本）将 TensorFlow/PyTorch 权重转换为 `.ckpt` 格式：

```bash
python tools/convert_bert_tf_to_ms.py --tf_ckpt_path ./bert_model --ms_ckpt_path ./bert_ms.ckpt
```

### 3. 设置模型路径

在 **init_parameter.py** 中修改：

```python
bert_model = "./bert_ms.ckpt"
```

---

## 🚀 Usage

### 运行完整实验

```bash
bash run.sh
```

### 加载 MTP 预训练模型

本仓库同时支持论文“MTP”预训练权重结构。若要开启：

```bash
bash run.sh --load_mtp
```

将你的 MTP checkpoint（MindSpore 格式）放置到指定目录，并在参数中提供路径。

### MTP 预训练说明

我们使用来自以下项目的 MTP 预训练权重，并在此基础上完成 step2 的进一步预训练：

[https://github.com/fanolabs/NID_ACLARR2022](https://github.com/fanolabs/NID_ACLARR2022)

已将其转换为 MindSpore 格式并适配到当前框架。

---

## 🧠 Model Architecture

我们更忠实地复现了原论文的方法，包括：

* 基于 BERT 的语义编码
* 借助概率建模进行意图发现
* 聚类 + 距离学习机制
* MTP 预训练模块（MindSpore 重构版）

模型整体结构如下：

![Model](./architecture.png)

---

## 🙏 Thanks & Acknowledgments

本项目的 MindSpore 实现参考了以下开源仓库的结构设计：

* [https://github.com/thuiar/DeepAligned-Clustering](https://github.com/thuiar/DeepAligned-Clustering)
* [https://github.com/fanolabs/NID_ACLARR2022](https://github.com/fanolabs/NID_ACLARR2022)

在此基础上进行了 MindSpore 的全量适配与性能优化。
