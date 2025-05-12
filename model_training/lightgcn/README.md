# LightGCN模型实现与训练

本目录包含LightGCN模型的实现，用于推荐系统的召回阶段。

## 文件结构

- `model.py`: LightGCN模型架构实现
- `utils.py`: 数据处理和评估工具函数
- `train.py`: 模型训练和评估脚本

## 模型架构

LightGCN是一种轻量级的基于图神经网络的协同过滤模型，专为推荐系统设计。它简化了传统GCN，去掉了转换矩阵和非线性激活函数，只保留了最核心的邻居聚合操作。

主要特点：
- 用户和物品被表示为同一个图的节点
- 用户-物品交互形成图的边
- 通过多层传播来学习用户和物品的嵌入
- 使用层间跳跃连接整合多层表示

## 训练方法

- 使用BPR (Bayesian Personalized Ranking) 损失函数
- 基于用户的历史交互采样正样本，随机采样负样本
- 支持多进程数据处理，加速样本生成
- 使用MLflow跟踪训练过程和指标

## 训练结果

在子集数据上（10000用户，5000物品）训练3个epoch后：
- Recall@5: 0.2232
- Recall@10: 0.3888
- Recall@20: 0.6208
- NDCG@10: 0.4164

## 使用方法

### 安装依赖

```bash
conda create -n recsys python=3.9
conda activate recsys
pip install -r ../requirements.txt
```

### 训练模型

```bash
python train.py --data_path ../../data_processing/lightgcn_data.csv --output_dir ../../model_output/lightgcn --n_epochs 5 --batch_size 64 --max_users 10000 --max_items 5000 --num_workers 8 --save_embeddings --experiment_name LightGCN_Test
```

参数说明：
- `--data_path`: 数据文件路径
- `--output_dir`: 模型输出路径
- `--n_epochs`: 训练轮数
- `--batch_size`: 批次大小
- `--max_users`: 最大用户数量，设为0表示使用所有用户
- `--max_items`: 最大物品数量，设为0表示使用所有物品
- `--num_workers`: 采样和数据加载的进程数
- `--save_embeddings`: 是否保存嵌入向量
- `--experiment_name`: MLflow实验名称

### 使用嵌入向量

训练后的嵌入向量保存在输出目录中：
- `user_embeddings.csv`: 用户嵌入向量
- `item_embeddings.csv`: 物品嵌入向量

可以基于这些嵌入向量计算用户与物品的相似度，用于召回阶段的推荐。

## 性能优化

- 使用多进程加速负样本采样
- GPU加速模型训练
- 支持早停以避免过拟合
- 支持数据子集训练，便于快速迭代实验 