# DIN (Deep Interest Network) 模型

本目录包含DIN模型的实现，用于推荐系统中的排序阶段。DIN模型通过注意力机制捕获用户对不同历史物品的兴趣强度，有效建模用户的动态兴趣变化。

## 模型架构

DIN模型的核心是基于注意力机制的用户兴趣表示：

1. **注意力层**：计算候选物品与用户历史物品的相关性权重，从而获取针对当前候选物品的用户兴趣表示。
2. **多层感知机**：将注意力加权后的用户兴趣向量、候选物品向量和历史物品平均向量拼接，通过多层神经网络预测用户对候选物品的点击概率。

## 文件说明

- `model.py`: DIN模型架构实现
- `utils.py`: 数据加载和处理工具
- `train.py`: 训练和评估脚本

## 数据格式

DIN模型需要以下两个数据文件：

1. **train_data.csv**: 包含用户历史行为和候选物品
   - `user_id`: 用户ID
   - `history_items`: 用户历史交互物品ID，以`|`分隔
   - `candidate_item`: 候选物品ID
   - `label`: 标签（1表示点击，0表示未点击）

2. **item_embeddings.csv**: 物品特征数据
   - `item_id`: 物品ID
   - `category_hash`、`brand_hash`、`price_scaled`等特征

## 训练命令

```bash
python train.py \
    --train_data_path ../../data_processing/train_data.csv \
    --item_embeddings_path ../../data_processing/item_embeddings.csv \
    --output_dir ../../model_output/din \
    --embedding_dim 64 \
    --attention_dim 64 \
    --mlp_hidden_dims 128,64,32 \
    --batch_size 1024 \
    --lr 0.001 \
    --n_epochs 20 \
    --device cuda
```

## 参数说明

- `--train_data_path`: 训练数据路径
- `--item_embeddings_path`: 物品特征数据路径
- `--output_dir`: 模型输出路径
- `--max_samples`: 最大样本数量（用于调试）
- `--embedding_dim`: 嵌入维度
- `--attention_dim`: 注意力层隐藏维度
- `--mlp_hidden_dims`: MLP隐藏层维度，用逗号分隔
- `--dropout`: Dropout率
- `--batch_size`: 批次大小
- `--lr`: 学习率
- `--weight_decay`: 权重衰减
- `--n_epochs`: 训练轮数
- `--early_stop`: 早停轮数
- `--num_workers`: 数据加载工作线程数
- `--experiment_name`: MLflow实验名称
- `--device`: 计算设备

## 模型输出

训练后，模型和相关配置将保存在`output_dir`目录中：

- `best_model.pth`：最佳模型参数和配置

## 性能指标

DIN模型使用以下指标评估性能：

- **AUC**：衡量模型区分正负样本的能力
- **Log Loss**：衡量预测概率的准确性
- **准确率**：使用0.5作为阈值的二分类准确率

## 训练过程

1. 加载训练数据和物品特征
2. 创建模型和优化器
3. 训练循环：
   - 前向传播计算损失
   - 反向传播更新参数
   - 验证集评估
   - 学习率调整
   - 早停检查
4. 保存最佳模型和配置

## MLflow集成

训练过程中，使用MLflow记录以下内容：
- 模型参数配置
- 训练和验证指标
- 最佳模型元数据

## 使用模型

```python
from model import DIN
from utils import load_model

# 加载模型
model = load_model(
    model_path='../../model_output/din/best_model.pth',
    item_feat_dim=3,  # 物品特征维度
    device='cuda'
)

# 预测
model.eval()
with torch.no_grad():
    preds = model(candidate_features, history_features, history_lengths)
```

## 注意事项

- 确保`history_items`字段中的物品ID都包含在`item_embeddings.csv`中
- 对于新物品，模型会使用默认的零向量表示
- 历史序列长度超过最大长度(默认50)时会被截断
- 在大数据集上训练时，建议调整`batch_size`和`num_workers`以适应可用内存 