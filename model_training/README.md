# 推荐系统模型训练模块

本模块负责训练和评估三阶段推荐系统架构中的各个模型组件：LightGCN（召回）、DIN（排序）和RankNet（重排序）。

## 目录结构

```
model_training/
├── README.md           # 本文件
├── requirements.txt    # 训练环境依赖
├── lightgcn/           # LightGCN 召回模型
│   ├── train.py        # 训练脚本
│   ├── model.py        # 模型定义
│   └── utils.py        # 工具函数
├── din/                # DIN 粗排模型
│   ├── train.py        # 训练脚本
│   ├── model.py        # 模型定义
│   └── utils.py        # 工具函数
├── ranknet/            # RankNet 精排模型
    ├── train.py        # 训练脚本
    ├── model.py        # 模型定义
    └── utils.py        # 工具函数
```

## 训练数据

训练使用的数据已经在 `data_processing` 文件夹中准备好：

- `lightgcn_data.csv`: 用于训练 LightGCN 召回模型的用户-物品交互数据
- `train_data.csv`: 用于训练 DIN 和 RankNet 的用户历史、候选物品及标签数据
- `item_embeddings.csv`: 物品特征数据，包含类别、品牌和价格特征

## 模型架构

### 1. LightGCN (召回模型)

- **输入**: 用户-物品交互图
- **输出**: 用户和物品的低维嵌入向量
- **目标**: 学习用户和物品的嵌入，以便可以通过向量相似度快速检索相关物品
- **优化指标**: 召回率 (Recall@K)、准确率 (Precision@K)
- **训练数据**: `lightgcn_data.csv`

### 2. DIN (粗排模型)

- **输入**: 用户历史行为序列、候选物品特征
- **输出**: 点击/兴趣概率
- **目标**: 对用户和物品的相关性进行精确建模，考虑用户的历史行为序列
- **优化指标**: AUC、Log Loss
- **训练数据**: `train_data.csv`, `item_embeddings.csv`

### 3. RankNet (精排模型)

- **输入**: 用户特征、物品特征、上下文特征
- **输出**: 物品间的相对排序得分
- **目标**: 学习物品间的相对优先级，以优化最终推荐列表
- **优化指标**: NDCG、MRR
- **训练数据**: `train_data.csv`, `item_embeddings.csv`

## 训练流程

1. **数据加载和预处理**
   - 读取CSV数据文件
   - 构建训练集和验证集
   - 数据转换为模型可用格式

2. **模型训练**
   - LightGCN: 学习用户和物品的嵌入
   - DIN: 使用用户历史行为序列和物品特征预测点击概率
   - RankNet: 学习物品间的相对排序关系

3. **模型评估**
   - 使用验证集评估模型性能
   - 记录关键指标：Recall@K, Precision@K, AUC, NDCG, MRR 等
   - 保存最佳模型

4. **模型导出**
   - 将训练好的模型保存为可用于服务的格式
   - 导出模型元数据，如特征映射、超参数等

## 模型服务集成

训练好的模型将被导出到一个统一的格式，以便API服务可以加载和使用。模型服务会使用这些导出的模型来进行实时推荐。

## 训练环境要求

- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+ (用于GPU训练)
- MLflow (用于实验跟踪和模型版本管理)

所有依赖项已在 `requirements.txt` 中列出。 