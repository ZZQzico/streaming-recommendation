import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import mlflow
from tqdm import tqdm
import random

class RankNetDataset(Dataset):
    """
    RankNet模型数据集
    处理用户物品对，构建正负样本对
    """
    def __init__(self, interactions, item_features, user_history_dict, neg_samples=1):
        """
        初始化数据集
        
        参数:
            interactions: 交互数据，包含用户、物品和标签
            item_features: 物品特征数据
            user_history_dict: 用户历史物品字典
            neg_samples: 每个正样本对应的负样本数量
        """
        self.interactions = interactions
        self.item_features = item_features
        self.user_history_dict = user_history_dict
        self.neg_samples = neg_samples
        
        # 物品特征维度
        self.item_feat_dim = len(self.item_features.columns) - 1  # 减去item_id列
        self.user_feat_dim = 128  # 用户特征维度，基于历史物品的平均嵌入
        
        # 创建物品ID到特征的映射
        self.item_features_dict = self._create_item_features_dict()
        
        # 提取并组织正样本
        self.positive_samples = self._extract_positive_samples()
        
        # 所有物品ID集合（用于负采样）
        self.all_item_ids = list(self.item_features_dict.keys())
        if 'UNK' in self.all_item_ids:
            self.all_item_ids.remove('UNK')
            
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
    
    def _extract_positive_samples(self):
        """提取正样本"""
        positive_samples = self.interactions[self.interactions['label'] == 1].copy()
        return positive_samples
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.positive_samples) * self.neg_samples
    
    def __getitem__(self, idx):
        """获取单个样本对（一个正样本和一个负样本）"""
        # 确定正样本索引
        pos_idx = idx // self.neg_samples
        pos_interaction = self.positive_samples.iloc[pos_idx]
        
        # 提取用户ID和正样本物品ID
        user_id = pos_interaction['user_id']
        pos_item_id = pos_interaction['candidate_item']
        
        # 获取用户的历史物品
        user_history = self.user_history_dict.get(user_id, [])
        
        # 获取负样本物品
        neg_item_id = self._sample_negative_item(user_id, user_history, pos_item_id)
        
        # 构建用户特征（基于历史物品的平均嵌入）
        user_features = self._build_user_features(user_history)
        
        # 获取物品特征
        pos_item_features = self._get_item_features(pos_item_id)
        neg_item_features = self._get_item_features(neg_item_id)
        
        return {
            'user_id': user_id,
            'user_features': user_features,
            'pos_item_features': pos_item_features,
            'neg_item_features': neg_item_features
        }
    
    def _build_user_features(self, history_items):
        """构建用户特征（基于历史物品的平均嵌入）"""
        if not history_items:
            # 如果没有历史记录，返回全0向量
            return np.zeros(self.user_feat_dim, dtype=np.float32)
        
        # 获取历史物品特征
        history_features = []
        for item_id in history_items:
            if item_id in self.item_features_dict:
                history_features.append(self.item_features_dict[item_id])
                
        if not history_features:
            return np.zeros(self.user_feat_dim, dtype=np.float32)
            
        # 计算平均特征
        history_features = np.array(history_features)
        avg_features = np.mean(history_features, axis=0)
        
        # 扩展到用户特征维度（如果需要）
        if len(avg_features) < self.user_feat_dim:
            padding = np.zeros(self.user_feat_dim - len(avg_features), dtype=np.float32)
            avg_features = np.concatenate([avg_features, padding])
        
        return avg_features
    
    def _get_item_features(self, item_id):
        """获取物品特征向量"""
        if item_id in self.item_features_dict:
            return self.item_features_dict[item_id]
        else:
            return self.item_features_dict['UNK']
    
    def _sample_negative_item(self, user_id, user_history, pos_item_id):
        """采样负样本物品"""
        # 用户交互过的物品（包括正样本）
        interacted_items = set(user_history + [pos_item_id])
        
        # 从所有物品中随机选择，直到找到一个未交互过的
        neg_item_id = random.choice(self.all_item_ids)
        attempts = 0
        max_attempts = 10  # 最大尝试次数
        
        while neg_item_id in interacted_items and attempts < max_attempts:
            neg_item_id = random.choice(self.all_item_ids)
            attempts += 1
            
        return neg_item_id

def load_data(train_data_path, item_embeddings_path, test_size=0.2, random_state=42, max_samples=None):
    """
    加载并处理RankNet模型训练数据
    
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
        user_feat_dim: 用户特征维度
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
    
    # 构建用户历史物品字典
    user_history_dict = {}
    for _, row in tqdm(interactions.iterrows(), desc="构建用户历史物品字典", total=len(interactions)):
        user_id = row['user_id']
        if pd.notna(row['history_items']) and row['history_items']:
            history_items = row['history_items'].split('|')
            user_history_dict[user_id] = history_items
        else:
            user_history_dict[user_id] = []
    
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
    
    # 物品特征维度和用户特征维度
    item_feat_dim = len(item_features.columns) - 1  # 减去item_id列
    user_feat_dim = 128  # 基于历史物品的平均嵌入
    
    # 划分训练集和验证集
    train_interactions, valid_interactions = train_test_split(
        interactions, test_size=test_size, random_state=random_state
    )
    
    print(f"训练集大小: {len(train_interactions)}, 验证集大小: {len(valid_interactions)}")
    
    # 创建数据集
    train_dataset = RankNetDataset(train_interactions, item_features, user_history_dict, neg_samples=5)
    valid_dataset = RankNetDataset(valid_interactions, item_features, user_history_dict, neg_samples=3)
    
    return train_dataset, valid_dataset, item_feat_dim, user_feat_dim

def collate_fn(batch):
    """
    数据批次整理函数
    
    参数:
        batch: 批次数据
        
    返回:
        整理后的批次数据
    """
    user_ids = []
    user_features = []
    pos_item_features = []
    neg_item_features = []
    
    for item in batch:
        user_ids.append(item['user_id'])
        user_features.append(item['user_features'])
        pos_item_features.append(item['pos_item_features'])
        neg_item_features.append(item['neg_item_features'])
    
    # 转换为张量
    user_features = torch.tensor(np.array(user_features), dtype=torch.float32)
    pos_item_features = torch.tensor(np.array(pos_item_features), dtype=torch.float32)
    neg_item_features = torch.tensor(np.array(neg_item_features), dtype=torch.float32)
    
    return {
        'user_ids': user_ids,
        'user_features': user_features,
        'pos_item_features': pos_item_features,
        'neg_item_features': neg_item_features
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
    评估模型
    
    参数:
        model: RankNet模型
        data_loader: 数据加载器
        device: 计算设备
        
    返回:
        metrics: 评估指标
    """
    model.eval()
    total_loss = 0
    total_samples = 0
    correct_pairs = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # 获取数据
            user_features = batch['user_features'].to(device)
            pos_item_features = batch['pos_item_features'].to(device)
            neg_item_features = batch['neg_item_features'].to(device)
            
            # 计算分数
            pos_scores = model(user_features, pos_item_features)
            neg_scores = model(user_features, neg_item_features)
            
            # 计算损失
            diff = pos_scores - neg_scores
            loss = torch.nn.functional.binary_cross_entropy_with_logits(diff, torch.ones_like(diff))
            
            # 统计
            batch_size = user_features.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # 计算正确预测的对数（正样本分数 > 负样本分数）
            correct_pairs += torch.sum(pos_scores > neg_scores).item()
    
    # 计算平均损失和准确率
    avg_loss = total_loss / total_samples
    accuracy = correct_pairs / total_samples
    
    # 计算NDCG和MRR
    # 注意：这里的计算是简化的，因为我们只有一个正样本和一个负样本
    ndcg = accuracy  # 在这种简化情况下，NDCG等同于准确率
    mrr = accuracy   # 同样，MRR也等同于准确率
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'ndcg': ndcg,
        'mrr': mrr
    }

def create_mlflow_experiment(experiment_name, tracking_uri=None):
    """
    创建MLflow实验
    
    参数:
        experiment_name: 实验名称
        tracking_uri: MLflow跟踪服务器地址
        
    返回:
        experiment_id: 实验ID
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    # 检查实验是否存在
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(experiment_name)
    
    return experiment_id

def save_model(model, save_path):
    """
    保存模型
    
    参数:
        model: RankNet模型
        save_path: 保存路径
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存模型
    model_state_dict = model.state_dict()
    model_config = {
        'user_feat_dim': model.user_embedding.in_features,
        'item_feat_dim': model.item_embedding.in_features,
        'embedding_dim': model.embedding_dim,
        'hidden_dims': [model.mlp[0].out_features, model.mlp[4].out_features, model.mlp[8].out_features]
    }
    
    torch.save({
        'model_state_dict': model_state_dict,
        'model_config': model_config
    }, save_path)
    
    print(f"模型已保存到: {save_path}")

def load_model(model_path, device):
    """
    加载模型
    
    参数:
        model_path: 模型路径
        device: 计算设备
        
    返回:
        model: 加载的模型
    """
    # 加载模型状态和配置
    checkpoint = torch.load(model_path, map_location=device)
    model_state_dict = checkpoint['model_state_dict']
    model_config = checkpoint['model_config']
    
    # 创建模型
    model = RankNet(
        user_feat_dim=model_config['user_feat_dim'],
        item_feat_dim=model_config['item_feat_dim'],
        embedding_dim=model_config['embedding_dim'],
        hidden_dims=model_config['hidden_dims']
    ).to(device)
    
    # 加载状态字典
    model.load_state_dict(model_state_dict)
    
    return model 