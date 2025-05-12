import os
import argparse
import time
import torch
import torch.optim as optim
import numpy as np
import mlflow
from tqdm import tqdm
import random
import pandas as pd

from model import LightGCN
from utils import (
    load_data, 
    create_user_item_graph, 
    sample_negative_items, 
    generate_candidates,
    calculate_metrics,
    create_mlflow_experiment
)

def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='LightGCN训练参数')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default='../data_processing/lightgcn_data.csv', help='训练数据路径')
    parser.add_argument('--output_dir', type=str, default='../model_output/lightgcn', help='模型输出路径')
    parser.add_argument('--test_size', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--max_users', type=int, default=50000, help='最大用户数量，设为0表示使用所有用户')
    parser.add_argument('--max_items', type=int, default=30000, help='最大物品数量，设为0表示使用所有物品')
    
    # 模型参数
    parser.add_argument('--embedding_dim', type=int, default=128, help='嵌入维度')
    parser.add_argument('--n_layers', type=int, default=3, help='图卷积层数')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    
    # 训练参数
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--n_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=2048, help='批次大小')
    parser.add_argument('--early_stop', type=int, default=10, help='早停轮数')
    parser.add_argument('--n_neg', type=int, default=1, help='每个正样本对应的负样本数量')
    parser.add_argument('--num_workers', type=int, default=0, help='采样和数据加载的进程数，0表示自动选择')
    
    # MLflow参数
    parser.add_argument('--experiment_name', type=str, default='LightGCN', help='MLflow实验名称')
    parser.add_argument('--tracking_uri', type=str, default=None, help='MLflow跟踪服务器地址')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='计算设备')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--save_embeddings', action='store_true', help='是否保存嵌入')
    
    return parser.parse_args()

def train(model, optimizer, train_loader, device, reg_weight=1e-5):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_reg_loss = 0.0
    total_samples = 0
    
    for batch in tqdm(train_loader, desc='Training'):
        user, pos_item, neg_item = batch
        user = user.to(device)
        pos_item = pos_item.to(device)
        neg_item = neg_item.to(device)
        
        # 计算BPR损失
        loss, reg_loss = model.bpr_loss(user, pos_item, neg_item)
        
        # 总损失 = BPR损失 + 正则化损失
        batch_loss = loss + reg_weight * reg_loss
        
        # 反向传播
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        
        # 累计统计
        batch_size = user.size(0)
        total_loss += loss.item() * batch_size
        total_reg_loss += reg_loss.item() * batch_size
        total_samples += batch_size
    
    # 计算平均损失
    avg_loss = float(total_loss / total_samples)
    avg_reg_loss = float(total_reg_loss / total_samples)
    
    return avg_loss, avg_reg_loss

def save_embeddings_fixed(model, user_mapping, item_mapping, save_dir):
    """
    保存训练好的用户和物品嵌入（修复版本）
    
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

def main():
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建MLflow实验
    experiment_id = create_mlflow_experiment(args.experiment_name, args.tracking_uri)
    
    # 确定最大用户数量和物品数量
    max_users = args.max_users if args.max_users > 0 else None
    max_items = args.max_items if args.max_items > 0 else None
    
    # 加载数据
    train_data, valid_data, user_mapping, item_mapping, n_users, n_items = load_data(
        args.data_path, 
        test_size=args.test_size, 
        random_state=args.seed,
        max_users=max_users,
        max_items=max_items
    )
    
    # 创建用户-物品二部图
    graph_data = create_user_item_graph(train_data, n_users, n_items)
    edge_index = graph_data.edge_index
    
    # 创建模型
    model = LightGCN(
        num_users=n_users,
        num_items=n_items,
        embedding_dim=args.embedding_dim,
        num_layers=args.n_layers,
        dropout=args.dropout
    )
    
    # 设置模型的边索引
    model.set_edge_index(edge_index)
    
    # 将模型移至指定设备
    model = model.to(args.device)
    edge_index = edge_index.to(args.device)
    
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 确定工作进程数
    num_workers = args.num_workers if args.num_workers > 0 else None
    dataloader_workers = max(4, args.num_workers) if args.num_workers > 0 else 4
    
    # 生成训练样本（正负样本对）
    print("生成训练样本...")
    user_indices, pos_item_indices, neg_item_indices = sample_negative_items(
        train_data, n_items, n_neg=args.n_neg, num_workers=num_workers
    )
    
    print(f"生成的训练样本数量: {len(user_indices)}")
    
    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(
        torch.LongTensor(user_indices),
        torch.LongTensor(pos_item_indices),
        torch.LongTensor(neg_item_indices)
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=dataloader_workers,
        pin_memory=True
    )
    
    # 为评估生成候选物品
    print("生成候选物品...")
    valid_users = valid_data['user_idx'].unique()
    all_item_indices = np.arange(n_items)
    user_candidates, user_positives = generate_candidates(
        valid_users, valid_data, all_item_indices, n_candidates=100
    )
    
    # 训练循环
    best_recall = 0
    best_epoch = 0
    early_stop_counter = 0
    
    with mlflow.start_run(experiment_id=experiment_id) as run:
        # 记录参数
        mlflow.log_params({
            'embedding_dim': args.embedding_dim,
            'n_layers': args.n_layers,
            'dropout': args.dropout,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'n_epochs': args.n_epochs,
            'batch_size': args.batch_size,
            'n_neg': args.n_neg,
            'n_users': n_users,
            'n_items': n_items,
            'max_users': args.max_users,
            'max_items': args.max_items,
            'num_workers': args.num_workers
        })
        
        print("开始训练...")
        for epoch in range(args.n_epochs):
            # 训练一个epoch
            start_time = time.time()
            train_loss, reg_loss = train(model, optimizer, train_loader, args.device)
            train_time = time.time() - start_time
            
            # 评估
            eval_users = list(user_candidates.keys())
            metrics = calculate_metrics(model, eval_users, user_candidates, user_positives, args.device)
            
            # 打印结果
            print(f"Epoch {epoch+1}/{args.n_epochs} - "
                  f"训练损失: {train_loss:.4f}, 正则化损失: {reg_loss:.4f}, "
                  f"耗时: {train_time:.2f}s, "
                  f"Recall@10: {metrics['recall_at_10']:.4f}, NDCG@10: {metrics['ndcg_at_10']:.4f}")
            
            # 确保所有指标都是标量float
            log_metrics = {
                'train_loss': float(train_loss),
                'reg_loss': float(reg_loss),
                'train_time': float(train_time)
            }
            
            # 添加评估指标
            for k, v in metrics.items():
                log_metrics[k] = float(v)
            
            # 记录指标
            mlflow.log_metrics(log_metrics, step=epoch)
            
            # 检查是否为最佳模型
            if metrics['recall_at_10'] > best_recall:
                best_recall = metrics['recall_at_10']
                best_epoch = epoch
                early_stop_counter = 0
                
                # 保存最佳模型
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
                
                # 记录最佳模型
                mlflow.pytorch.log_model(model, "best_model")
            else:
                early_stop_counter += 1
                
            # 早停
            if early_stop_counter >= args.early_stop:
                print(f"早停: 在{args.early_stop}个epoch内未见改善")
                break
        
        # 加载最佳模型
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pth')))
        
        # 记录最佳指标
        final_metrics = calculate_metrics(model, eval_users, user_candidates, user_positives, args.device)
        print(f"最佳模型 (Epoch {best_epoch+1}):")
        for k, v in final_metrics.items():
            print(f"  {k}: {v:.4f}")
        
        # 确保所有指标都是标量float
        best_metrics = {}
        for k, v in final_metrics.items():
            best_metrics[f'best_{k}'] = float(v)
        
        mlflow.log_metrics(best_metrics)
        
        # 保存嵌入
        if args.save_embeddings:
            save_embeddings_fixed(model, user_mapping, item_mapping, args.output_dir)
    
    print("训练完成!")

if __name__ == '__main__':
    main() 