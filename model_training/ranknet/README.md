# RankNet 模型实现

RankNet 是一个基于神经网络的成对排序算法，用于学习物品间的相对优先级，从而优化最终的推荐列表。在我们的推荐系统中，它作为最后一级精排模型使用。

## 模型架构

RankNet 的主要特点是使用成对（pairwise）比较来学习物品的排序关系。对于每个用户，模型学习正样本物品应该比负样本物品具有更高的排名。

核心组件包括：

1. **用户特征嵌入层**：将用户特征向量转换为低维密集表示
2. **物品特征嵌入层**：将物品特征向量转换为低维密集表示
3. **多层感知机（MLP）**：处理用户和物品嵌入的拼接特征
4. **排序评分输出层**：输出物品的排序分数

## 训练数据

RankNet 使用以下数据进行训练：

- `train_data.csv`：包含用户ID、历史物品序列、候选物品和标签（正/负）
- `item_embeddings.csv`：包含物品ID和对应的特征向量（类别、品牌、价格等）

## 损失函数

RankNet 使用交叉熵损失函数来优化成对排序目标：

```
loss = -log(σ(s_i - s_j))
```

其中：
- s_i 是正样本物品的得分
- s_j 是负样本物品的得分
- σ 是 sigmoid 函数

## 评估指标

主要评估指标包括：

- **NDCG (Normalized Discounted Cumulative Gain)**：评估排序质量
- **MRR (Mean Reciprocal Rank)**：评估第一个相关结果的位置
- **准确率 (Accuracy)**：正确排序的成对比例

## 使用方法

### 训练模型

```bash
python train.py \
    --train_data_path ../../data_processing/train_data.csv \
    --item_embeddings_path ../../data_processing/item_embeddings.csv \
    --output_dir ../../model_output/ranknet \
    --embedding_dim 128 \
    --hidden_dims 128,64,32 \
    --batch_size 1024 \
    --n_epochs 20 \
    --device cuda
```

### 参数说明

#### 数据参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--train_data_path` | 训练数据路径 | `../../data_processing/train_data.csv` |
| `--item_embeddings_path` | 物品特征数据路径 | `../../data_processing/item_embeddings.csv` |
| `--output_dir` | 模型输出路径 | `../../model_output/ranknet` |
| `--max_samples` | 最大样本数量，用于调试 | 0 (使用全部数据) |

#### 模型参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--embedding_dim` | 嵌入维度 | 128 |
| `--hidden_dims` | MLP隐藏层维度，用逗号分隔 | `128,64,32` |
| `--dropout` | Dropout率 | 0.2 |

#### 训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--batch_size` | 批次大小 | 1024 |
| `--lr` | 学习率 | 0.001 |
| `--weight_decay` | 权重衰减 | 1e-5 |
| `--n_epochs` | 训练轮数 | 20 |
| `--early_stop` | 早停轮数 | 5 |
| `--num_workers` | 数据加载工作线程数 | 4 |

#### MLflow参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--experiment_name` | MLflow实验名称 | `RankNet` |
| `--tracking_uri` | MLflow跟踪服务器地址 | `None` |

#### 其他参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--device` | 计算设备 | `cuda` 或 `cpu` |
| `--seed` | 随机种子 | 42 |

## 模型集成

在推荐系统中，RankNet 作为最后一级精排模型，用于对前面阶段（召回和粗排）产生的候选物品进行最终排序。集成流程如下：

1. **LightGCN**：从海量物品中召回数百个相关物品
2. **DIN**：对召回的物品进行初步排序
3. **RankNet**：对初步排序的TOP-N物品进行精排，生成最终推荐列表

## 实现细节

RankNet 的核心实现包括以下文件：

- `model.py`：RankNet 模型定义
- `utils.py`：数据处理和辅助功能
- `train.py`：训练和评估逻辑

模型训练过程使用 MLflow 进行实验跟踪，记录超参数、训练指标和模型文件。 