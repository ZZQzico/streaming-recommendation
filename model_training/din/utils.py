import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import mlflow
from tqdm import tqdm

class DINDataset(Dataset):
    """
    DIN模型数据集
    处理用户历史物品序列和候选物品
    """
    def __init__(self, interactions, item_features, max_seq_len=50):
        """
        初始化数据集
        
        参数:
            interactions: 交互数据，包含用户历史和候选物品
            item_features: 物品特征数据
            max_seq_len: 历史序列最大长度
        """
        self.interactions = interactions
        self.item_features = item_features
        self.max_seq_len = max_seq_len
        
        # 物品特征维度
        self.item_feat_dim = len(self.item_features.columns) - 1  # 减去item_id列
        
        # 创建物品ID到特征的映射
        self.item_features_dict = self._create_item_features_dict()
        
    def _create_item_features_dict(self):
        """创建物品ID到特征向量的映射"""
        item_features_dict = {}
        
        for _, row in self.item_features.iterrows():
            item_id = row['item_id']
            features = row.drop('item_id').values.astype(np.float32)
            item_features_dict[item_id] = features
            
        # 创建一个默认特征向量用于未知物品
        default_features = np.zeros(self.item_feat_dim, dtype=np.float32)
        item_features_dict['UNK'] = default_features
        
        return item_features_dict
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.interactions)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        interaction = self.interactions.iloc[idx]
        
        # 提取用户ID、候选物品ID和标签
        user_id = interaction['user_id']
        candidate_item = interaction['candidate_item']
        label = interaction['label']
        
        # 获取历史物品列表
        if pd.notna(interaction['history_items']) and interaction['history_items']:
            history_items = interaction['history_items'].split('|')
        else:
            history_items = []
        
        # 截断或填充历史序列至max_seq_len
        if len(history_items) > self.max_seq_len:
            # 保留最近的max_seq_len个物品
            history_items = history_items[-self.max_seq_len:]
        
        # 历史序列实际长度
        history_length = len(history_items)
        
        # 填充历史序列至max_seq_len
        history_items = history_items + ['UNK'] * (self.max_seq_len - len(history_items))
        
        # 获取候选物品特征
        candidate_features = self._get_item_features(candidate_item)
        
        # 获取历史物品特征
        history_features = np.array([self._get_item_features(item) for item in history_items])
        
        return {
            'user_id': user_id,
            'candidate_features': candidate_features,
            'history_features': history_features,
            'history_length': history_length,
            'label': label
        }
    
    def _get_item_features(self, item_id):
        """获取物品特征向量"""
        if item_id in self.item_features_dict:
            return self.item_features_dict[item_id]
        else:
            return self.item_features_dict['UNK']

def load_data(train_data_path, item_embeddings_path, test_size=0.2, random_state=42, max_samples=None):
    """
    加载并处理DIN模型训练数据
    
    参数:
        train_data_path: 训练数据路径
        item_embeddings_path: 物品特征数据路径
        test_size: 验证集比例
        random_state: 随机种子
        max_samples: 最大样本数量，用于调试
        
    返回:
        train_dataset: 训练数据集
        valid_dataset: 验证数据集
        item_feat_dim: 物品特征维度
    """
    print(f"加载训练数据: {train_data_path}")
    
    # 加载交互数据
    interactions = pd.read_csv(train_data_path)
    
    # 限制样本数量（用于调试）
    if max_samples and max_samples > 0:
        interactions = interactions.sample(min(max_samples, len(interactions)), random_state=random_state)
    
    print(f"交互数据大小: {len(interactions)}")
    
    # 提取交互数据中涉及到的所有物品ID
    print("提取需要的物品ID...")
    candidate_items = set(interactions['candidate_item'].unique())
    
    # 处理历史物品列表
    history_items = set()
    for hist in tqdm(interactions['history_items'].dropna(), desc="处理历史物品"):
        if isinstance(hist, str) and hist:
            items = hist.split('|')
            history_items.update(items)
    
    # 合并所有需要的物品ID
    needed_items = candidate_items.union(history_items)
    print(f"需要加载的物品特征数量: {len(needed_items)}")
    
    # 加载物品特征
    print(f"加载物品特征: {item_embeddings_path}")
    
    # 使用分块读取和过滤，避免一次性加载全部数据
    chunk_size = 500000  # 每次读取的行数
    filtered_items = []
    
    print("分块读取物品特征...")
    for chunk in tqdm(pd.read_csv(item_embeddings_path, chunksize=chunk_size), desc="过滤物品特征"):
        # 只保留需要的物品
        filtered_chunk = chunk[chunk['item_id'].isin(needed_items)]
        filtered_items.append(filtered_chunk)
        
        # 如果已经找到全部需要的物品，可以提前退出
        if len(filtered_items) > 0 and len(pd.concat(filtered_items)['item_id'].unique()) >= len(needed_items):
            break
    
    # 合并所有过滤后的数据块
    item_features = pd.concat(filtered_items, ignore_index=True)
    
    # 添加UNK物品
    if 'UNK' not in item_features['item_id'].values:
        # 创建UNK物品特征（全为0）
        unk_features = pd.DataFrame({
            'item_id': ['UNK'],
            'category_hash': [0.0],
            'brand_hash': [0.0],
            'price_scaled': [0.0]
        })
        item_features = pd.concat([item_features, unk_features], ignore_index=True)
    
    print(f"过滤后的物品特征数据大小: {len(item_features)}")
    
    # 物品特征维度
    item_feat_dim = len(item_features.columns) - 1  # 减去item_id列
    
    # 划分训练集和验证集
    train_interactions, valid_interactions = train_test_split(
        interactions, test_size=test_size, random_state=random_state
    )
    
    print(f"训练集大小: {len(train_interactions)}, 验证集大小: {len(valid_interactions)}")
    
    # 创建数据集
    train_dataset = DINDataset(train_interactions, item_features)
    valid_dataset = DINDataset(valid_interactions, item_features)
    
    return train_dataset, valid_dataset, item_feat_dim

def collate_fn(batch):
    """
    数据批次整理函数
    
    参数:
        batch: 批次数据
        
    返回:
        整理后的批次数据
    """
    user_ids = [item['user_id'] for item in batch]
    candidate_features = np.array([item['candidate_features'] for item in batch])
    history_features = np.array([item['history_features'] for item in batch])
    history_lengths = np.array([item['history_length'] for item in batch])
    labels = np.array([item['label'] for item in batch])
    
    return {
        'user_ids': user_ids,
        'candidate_features': torch.FloatTensor(candidate_features),
        'history_features': torch.FloatTensor(history_features),
        'history_lengths': torch.LongTensor(history_lengths),
        'labels': torch.FloatTensor(labels)
    }

def create_data_loaders(train_dataset, valid_dataset, batch_size, num_workers=4):
    """
    创建数据加载器
    
    参数:
        train_dataset: 训练数据集
        valid_dataset: 验证数据集
        batch_size: 批次大小
        num_workers: 工作线程数
        
    返回:
        train_loader: 训练数据加载器
        valid_loader: 验证数据加载器
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, valid_loader

def evaluate_model(model, data_loader, device):
    """
    评估模型性能
    
    参数:
        model: DIN模型
        data_loader: 数据加载器
        device: 计算设备
        
    返回:
        metrics: 评估指标字典
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            candidate_features = batch['candidate_features'].to(device)
            history_features = batch['history_features'].to(device)
            history_lengths = batch['history_lengths'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            preds = model(candidate_features, history_features, history_lengths)
            
            # 计算损失
            loss = model.calculate_loss(candidate_features, history_features, history_lengths, labels)
            total_loss += loss.item() * len(labels)
            
            # 收集预测和标签
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算评估指标
    from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 二分类阈值
    binary_preds = (all_preds >= 0.5).astype(int)
    
    # 计算AUC
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except:
        auc = 0.5  # 如果所有标签都是同一个类，AUC计算会出错
    
    # 计算Log Loss
    logloss = log_loss(all_labels, all_preds)
    
    # 计算准确率
    accuracy = accuracy_score(all_labels, binary_preds)
    
    # 计算平均损失
    avg_loss = total_loss / len(all_labels)
    
    metrics = {
        'loss': avg_loss,
        'auc': auc,
        'logloss': logloss,
        'accuracy': accuracy
    }
    
    return metrics

def create_mlflow_experiment(experiment_name, tracking_uri=None):
    """
    创建或获取MLflow实验
    
    参数:
        experiment_name: 实验名称
        tracking_uri: MLflow跟踪服务器地址
        
    返回:
        experiment_id: 实验ID
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    # 获取或创建实验
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(experiment_name)
    
    return experiment_id

def save_model(model, save_path):
    """
    保存模型及其配置
    
    参数:
        model: 模型对象
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存模型参数
    torch.save({
        'state_dict': model.state_dict(),
        'config': {
            'embedding_dim': model.embedding_dim,
            'attention_dim': model.attention.attention_size,
            'mlp_hidden_dims': [layer.out_features for layer in model.mlp if isinstance(layer, nn.Linear)],
        }
    }, save_path)
    
    print(f"模型已保存至 {save_path}")

def load_model(model_path, item_feat_dim, device):
    """
    加载保存的模型
    
    参数:
        model_path: 模型文件路径
        item_feat_dim: 物品特征维度
        device: 计算设备
        
    返回:
        model: 加载的模型
    """
    from model import DIN
    
    # 加载模型参数
    checkpoint = torch.load(model_path, map_location=device)
    
    # 获取模型配置
    config = checkpoint['config']
    
    # 创建模型
    model = DIN(
        item_feat_dim=item_feat_dim,
        embedding_dim=config['embedding_dim'],
        attention_dim=config['attention_dim'],
        mlp_hidden_dims=config['mlp_hidden_dims']
    ).to(device)
    
    # 加载模型参数
    model.load_state_dict(checkpoint['state_dict'])
    
    return model 