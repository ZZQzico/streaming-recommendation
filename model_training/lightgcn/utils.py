import os
import pdb
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import multiprocessing as mp
from functools import partial

def load_data(data_path, test_size=0.2, random_state=42, max_users=None, max_items=None):
    """
    加载数据并进行训练集和验证集划分，可以限制用户和物品数量
    
    参数:
        data_path: 数据文件路径
        test_size: 验证集比例
        random_state: 随机种子
        max_users: 最大用户数量，如果为None则使用所有用户
        max_items: 最大物品数量，如果为None则使用所有物品
        
    返回:
        train_data: 训练数据
        valid_data: 验证数据
        user_mapping: 用户ID到索引的映射
        item_mapping: 物品ID到索引的映射
        n_users: 用户数量
        n_items: 物品数量
    """
    print(f"加载数据: {data_path}")
    df = pd.read_csv(data_path)
    print(f"原始数据大小: {len(df)}")
    
    # 限制用户数量
    if max_users is not None and max_users > 0:
        unique_users = df['user_id'].value_counts().index[:max_users]
        df = df[df['user_id'].isin(unique_users)]
        print(f"限制后的数据大小: {len(df)}")
    else:
        unique_users = df['user_id'].unique()
    
    # 限制物品数量
    if max_items is not None and max_items > 0:
        unique_items = df['item_id'].value_counts().index[:max_items]
        df = df[df['item_id'].isin(unique_items)]
        print(f"限制后的数据大小: {len(df)}")
    else:
        unique_items = df['item_id'].unique()
    
    # 由于可能筛选了用户和物品，重新获取唯一值
    unique_users = df['user_id'].unique()
    unique_items = df['item_id'].unique()
    
    user_mapping = {user: idx for idx, user in enumerate(unique_users)}
    item_mapping = {item: idx for idx, item in enumerate(unique_items)}
    
    n_users = len(user_mapping)
    n_items = len(item_mapping)
    
    print(f"用户数量: {n_users}, 物品数量: {n_items}")
    
    # 将原始ID转换为索引
    df['user_idx'] = df['user_id'].map(user_mapping)
    df['item_idx'] = df['item_id'].map(item_mapping)
    
    # 划分训练集和验证集
    train_data, valid_data = train_test_split(df, test_size=test_size, random_state=random_state)
    
    print(f"训练集大小: {len(train_data)}, 验证集大小: {len(valid_data)}")
    
    return train_data, valid_data, user_mapping, item_mapping, n_users, n_items

def create_user_item_graph(train_data, n_users, n_items):
    """
    创建用户-物品二部图
    
    参数:
        train_data: 训练数据
        n_users: 用户数量
        n_items: 物品数量
        
    返回:
        data: PyTorch Geometric图数据对象
    """
    # 创建边索引
    user_indices = train_data['user_idx'].values
    item_indices = train_data['item_idx'].values + n_users  # 物品索引从 n_users 开始
    
    # 创建双向边 (用户->物品, 物品->用户)
    edge_index = torch.tensor([
        np.concatenate([user_indices, item_indices]),
        np.concatenate([item_indices, user_indices])
    ], dtype=torch.long)
    
    # 创建图数据对象
    data = Data(edge_index=edge_index, num_nodes=n_users + n_items)
    
    return data

def generate_candidates(user_ids, valid_data, item_indices, n_candidates=100):
    """
    为每个用户生成用于评估的候选物品集合
    
    参数:
        user_ids: 用户ID列表
        valid_data: 验证数据
        item_indices: 所有物品索引
        n_candidates: 每个用户的候选物品数量
        
    返回:
        user_candidates: 每个用户的候选物品集合
        user_positives: 每个用户的正样本物品
    """
    user_candidates = {}
    user_positives = {}
    
    for user_id in user_ids:
        # 找出用户的正样本物品
        user_pos = valid_data[valid_data['user_idx'] == user_id]['item_idx'].values
        
        # 如果没有正样本，跳过该用户
        if len(user_pos) == 0:
            continue
            
        # 找出用户的负样本物品（随机选择）
        user_neg = list(set(item_indices) - set(user_pos))
        
        # 计算需要的负样本数量
        n_neg_needed = n_candidates - len(user_pos)
        
        # 确保我们不要求超过可用的负样本数量
        if n_neg_needed <= 0:
            # 如果正样本数量已经超过或等于n_candidates，则只使用正样本
            candidates = user_pos[:n_candidates]
            user_candidates[user_id] = candidates
            user_positives[user_id] = user_pos
            continue
        
        # 确保采样数量不超过可用数量
        n_neg_to_sample = min(n_neg_needed, len(user_neg))
        
        if n_neg_to_sample <= 0 or len(user_neg) == 0:
            # 如果没有足够的负样本，则只使用正样本
            candidates = user_pos
        else:
            # 采样负样本并合并
            neg_samples = random.sample(user_neg, n_neg_to_sample)
            candidates = np.concatenate([user_pos, neg_samples])
        
        user_candidates[user_id] = candidates
        user_positives[user_id] = user_pos
    
    return user_candidates, user_positives

def calculate_metrics(model, user_ids, user_candidates, user_positives, device, k_list=[5, 10, 20]):
    """
    计算推荐系统评估指标
    
    参数:
        model: LightGCN模型
        user_ids: 用户ID列表
        user_candidates: 每个用户的候选物品集合
        user_positives: 每个用户的正样本物品
        device: 计算设备
        k_list: 计算指标的K值列表
        
    返回:
        metrics: 包含各项指标的字典
    """
    model.eval()
    
    # 存储不同K值的指标
    precision = {k: [] for k in k_list}
    recall = {k: [] for k in k_list}
    ndcg = {k: [] for k in k_list}
    
    # 首先获取所有嵌入，避免重复计算
    with torch.no_grad():
        edge_index = model.edge_index.to(device)
        user_emb, item_emb = model.forward(edge_index)
        
        for user_id in tqdm(user_ids, desc="Evaluating"):
            # 如果用户没有候选物品，跳过
            if user_id not in user_candidates:
                continue
                
            candidates = user_candidates[user_id]
            positives = user_positives[user_id]
            
            # 获取用户和候选物品嵌入
            user_emb_i = user_emb[user_id].unsqueeze(0)  # [1, embedding_dim]
            item_embs = item_emb[candidates]  # [n_candidates, embedding_dim]
            
            # 计算预测评分
            scores = torch.matmul(user_emb_i, item_embs.t()).squeeze()  # [n_candidates]
            
            # 获取评分排名最高的物品索引
            _, indices = torch.sort(scores, descending=True)
            
            # 计算各项指标
            recommended_items = candidates[indices.cpu().numpy()]
            
            for k in k_list:
                top_k_items = recommended_items[:k]
                
                # 计算Precision@K
                n_relevant = len(set(top_k_items) & set(positives))
                precision[k].append(n_relevant / k)
                
                # 计算Recall@K
                recall[k].append(n_relevant / len(positives))
                
                # 计算NDCG@K
                idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(positives), k)))
                dcg = 0.0
                for i, item in enumerate(top_k_items):
                    if item in positives:
                        dcg += 1.0 / np.log2(i + 2)
                
                ndcg[k].append(dcg / idcg if idcg > 0 else 0)
    
    # 计算平均指标，使用MLflow兼容的名称
    metrics = {}
    for k in k_list:
        metrics[f'precision_at_{k}'] = np.mean(precision[k])
        metrics[f'recall_at_{k}'] = np.mean(recall[k])
        metrics[f'ndcg_at_{k}'] = np.mean(ndcg[k])
    
    return metrics

def save_embeddings(model, user_mapping, item_mapping, save_dir):
    """
    保存训练好的用户和物品嵌入
    
    参数:
        model: 训练好的LightGCN模型
        user_mapping: 用户ID到索引的映射
        item_mapping: 物品ID到索引的映射
        save_dir: 保存路径
    """
    model.eval()
    
    # 获取最终的用户和物品嵌入
    with torch.no_grad():
        device = next(model.parameters()).device
        edge_index = model.edge_index.to(device)
        user_emb, item_emb = model.forward(edge_index)
    
    # 创建反向映射
    idx_to_user = {idx: user for user, idx in user_mapping.items()}
    idx_to_item = {idx: item for item, idx in item_mapping.items()}
    
    # 保存用户嵌入
    user_emb_df = pd.DataFrame()
    user_emb_df['user_id'] = [idx_to_user[i] for i in range(len(user_mapping))]
    
    for i in range(user_emb.shape[1]):
        user_emb_df[f'emb_{i}'] = user_emb[:, i].cpu().numpy()
    
    # 保存物品嵌入
    item_emb_df = pd.DataFrame()
    item_emb_df['item_id'] = [idx_to_item[i] for i in range(len(item_mapping))]
    
    for i in range(item_emb.shape[1]):
        item_emb_df[f'emb_{i}'] = item_emb[:, i].cpu().numpy()
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存为CSV文件
    user_emb_df.to_csv(os.path.join(save_dir, 'user_embeddings.csv'), index=False)
    item_emb_df.to_csv(os.path.join(save_dir, 'item_embeddings.csv'), index=False)
    
    print(f"嵌入已保存至 {save_dir}")

def _sample_negatives_for_user(user_data, all_item_indices, n_neg):
    """
    为单个用户采样负样本的辅助函数（用于多进程）
    
    参数:
        user_data: 包含用户ID和正样本列表的元组 (user_id, pos_item_list)
        all_item_indices: 所有物品的索引集合
        n_neg: 每个正样本对应的负样本数量
    
    返回:
        users, pos_items, neg_items: 用户、正样本和负样本的列表
    """
    user, pos_item_list = user_data
    pos_item_set = set(pos_item_list)
    
    # 找出用户没有交互过的物品作为负样本集
    pdb.set_trace()
    user_neg_item_pool = list(all_item_indices - pos_item_set)
    
    # 计算能够采样的负样本数量
    neg_item_count = min(n_neg, len(user_neg_item_pool))
    if neg_item_count <= 0 or not pos_item_list:
        return [], [], []
    
    # 计算总共需要的负样本数量
    total_neg_samples = len(pos_item_list) * neg_item_count
    
    # 如果负样本池不足，则采样可能的最大数量
    if total_neg_samples > len(user_neg_item_pool):
        # 从用户的负样本池中随机采样所需的负样本(有放回)
        sampled_neg_items = np.random.choice(user_neg_item_pool, total_neg_samples, replace=True)
    else:
        # 随机采样负样本(无放回)
        sampled_neg_items = np.random.choice(user_neg_item_pool, total_neg_samples, replace=False)
    
    # 构建用户索引和正样本索引数组
    users = np.repeat(user, total_neg_samples)
    
    # 为每个正样本重复neg_item_count次
    pos_items = np.repeat(pos_item_list, neg_item_count)
    
    return users.tolist(), pos_items.tolist(), sampled_neg_items.tolist()

def sample_negative_items(train_data, n_items, n_neg=1, num_workers=None):
    """
    为每个正样本采样对应的负样本，使用多进程加速
    
    参数:
        train_data: 训练数据
        n_items: 物品总数
        n_neg: 每个正样本对应的负样本数量
        num_workers: 进程数量，默认为CPU核心数的一半
        
    返回:
        user_indices: 用户索引
        pos_item_indices: 正样本物品索引
        neg_item_indices: 负样本物品索引
    """
    print(f"使用多进程生成训练样本...")
    
    # 如果未指定进程数，使用CPU核心数的一半
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() // 2)
    
    print(f"使用 {num_workers} 个进程")
    
    # 按用户分组数据
    user_pos_items = train_data.groupby('user_idx')['item_idx'].apply(list).to_dict()
    
    all_item_indices = set(range(n_items))
    
    # 准备多进程的输入数据
    user_data_list = list(user_pos_items.items())
    
    # 使用多进程加速采样过程
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(
                partial(_sample_negatives_for_user, all_item_indices=all_item_indices, n_neg=n_neg),
                user_data_list
            ),
            total=len(user_data_list),
            desc="Sampling negatives"
        ))
    
    # 合并所有结果
    all_users = []
    all_pos_items = []
    all_neg_items = []
    
    for users, pos_items, neg_items in results:
        all_users.extend(users)
        all_pos_items.extend(pos_items)
        all_neg_items.extend(neg_items)
    
    return np.array(all_users), np.array(all_pos_items), np.array(all_neg_items)

def create_mlflow_experiment(experiment_name, tracking_uri=None):
    """
    创建MLflow实验
    
    参数:
        experiment_name: 实验名称
        tracking_uri: MLflow跟踪服务器地址
        
    返回:
        experiment_id: 实验ID
    """
    import mlflow
    
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    # 获取或创建实验
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    
    return experiment_id 