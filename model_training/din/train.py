import os
import argparse
import torch
import torch.optim as optim
import numpy as np
import mlflow
from tqdm import tqdm
import random
from model import DIN
from utils import (
    load_data,
    create_data_loaders,
    evaluate_model,
    create_mlflow_experiment,
    save_model
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
    parser = argparse.ArgumentParser(description='DIN训练参数')
    
    # 数据参数
    parser.add_argument('--train_data_path', type=str, default='../../data_processing/train_data.csv', help='训练数据路径')
    parser.add_argument('--item_embeddings_path', type=str, default='../../data_processing/item_embeddings.csv', help='物品特征数据路径')
    parser.add_argument('--output_dir', type=str, default='../../model_output/din', help='模型输出路径')
    parser.add_argument('--max_samples', type=int, default=0, help='最大样本数量，用于调试，设为0表示使用所有样本')
    
    # 模型参数
    parser.add_argument('--embedding_dim', type=int, default=128, help='嵌入维度')
    parser.add_argument('--attention_dim', type=int, default=128, help='注意力层隐藏维度')
    parser.add_argument('--mlp_hidden_dims', type=str, default='128,64,32', help='MLP隐藏层维度，用逗号分隔')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout率')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=1024, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--n_epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--early_stop', type=int, default=100, help='早停轮数')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载工作线程数')
    
    # MLflow参数
    parser.add_argument('--experiment_name', type=str, default='DIN', help='MLflow实验名称')
    parser.add_argument('--tracking_uri', type=str, default=None, help='MLflow跟踪服务器地址')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='计算设备')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    return parser.parse_args()

def train_epoch(model, optimizer, train_loader, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_samples = 0
    
    for batch in tqdm(train_loader, desc='Training'):
        # 获取数据
        candidate_features = batch['candidate_features'].to(device)
        history_features = batch['history_features'].to(device)
        history_lengths = batch['history_lengths'].to(device)
        labels = batch['labels'].to(device)
        
        # 清除梯度
        optimizer.zero_grad()
        
        # 计算损失
        loss = model.calculate_loss(candidate_features, history_features, history_lengths, labels)
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        # 统计
        batch_size = candidate_features.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    
    # 计算平均损失
    avg_loss = total_loss / total_samples
    
    return avg_loss

def validate(model, valid_loader, device):
    """验证模型"""
    metrics = evaluate_model(model, valid_loader, device)
    return metrics

def main():
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建MLflow实验
    experiment_id = create_mlflow_experiment(args.experiment_name, args.tracking_uri)
    
    # 加载数据
    max_samples = args.max_samples if args.max_samples > 0 else None
    train_dataset, valid_dataset, item_feat_dim = load_data(
        args.train_data_path, 
        args.item_embeddings_path,
        max_samples=max_samples
    )
    
    # 创建数据加载器
    train_loader, valid_loader = create_data_loaders(
        train_dataset, 
        valid_dataset, 
        args.batch_size, 
        args.num_workers
    )
    
    # 解析MLP隐藏层维度
    mlp_hidden_dims = [int(dim) for dim in args.mlp_hidden_dims.split(',')]
    
    # 创建模型
    model = DIN(
        item_feat_dim=item_feat_dim,
        embedding_dim=args.embedding_dim,
        attention_dim=args.attention_dim,
        mlp_hidden_dims=mlp_hidden_dims,
        dropout=args.dropout
    ).to(args.device)
    
    # 创建优化器
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # 创建学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=2, 
        verbose=True
    )
    
    # 训练循环
    best_auc = 0
    best_epoch = 0
    early_stop_counter = 0
    
    with mlflow.start_run(experiment_id=experiment_id) as run:
        # 记录参数
        mlflow.log_params({
            'embedding_dim': args.embedding_dim,
            'attention_dim': args.attention_dim,
            'mlp_hidden_dims': args.mlp_hidden_dims,
            'dropout': args.dropout,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'n_epochs': args.n_epochs,
            'item_feat_dim': item_feat_dim,
            'train_samples': len(train_dataset),
            'valid_samples': len(valid_dataset)
        })
        
        for epoch in range(args.n_epochs):
            # 训练
            train_loss = train_epoch(model, optimizer, train_loader, args.device)
            
            # 验证
            valid_metrics = validate(model, valid_loader, args.device)
            valid_loss = valid_metrics['loss']
            valid_auc = valid_metrics['auc']
            valid_logloss = valid_metrics['logloss']
            valid_accuracy = valid_metrics['accuracy']
            
            # 更新学习率
            scheduler.step(valid_loss)
            
            # 打印指标
            print(f"Epoch {epoch+1}/{args.n_epochs} - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Valid Loss: {valid_loss:.4f}, "
                  f"Valid AUC: {valid_auc:.4f}, "
                  f"Valid LogLoss: {valid_logloss:.4f}, "
                  f"Valid Accuracy: {valid_accuracy:.4f}")
            
            # 记录指标
            mlflow.log_metrics({
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'valid_auc': valid_auc,
                'valid_logloss': valid_logloss,
                'valid_accuracy': valid_accuracy
            }, step=epoch)
            
            # 检查是否是最佳模型
            if valid_auc > best_auc:
                best_auc = valid_auc
                best_epoch = epoch
                early_stop_counter = 0
                
                # 保存最佳模型
                model_path = os.path.join(args.output_dir, 'best_model.pth')
                save_model(model, model_path)
                
                print(f"保存最佳模型，验证AUC: {valid_auc:.4f}")
            else:
                early_stop_counter += 1
                
            # 早停
            if early_stop_counter >= args.early_stop:
                print(f"早停触发，最佳模型出现在Epoch {best_epoch+1}")
                break
        
        # 记录最佳性能
        mlflow.log_metrics({
            'best_auc': best_auc,
            'best_epoch': best_epoch + 1
        })
        
        print(f"训练完成，最佳验证AUC: {best_auc:.4f}, 最佳Epoch: {best_epoch+1}")
        
        # 保存嵌入和模型配置信息
        model_meta = {
            'model_type': 'DIN',
            'embedding_dim': args.embedding_dim,
            'attention_dim': args.attention_dim,
            'mlp_hidden_dims': mlp_hidden_dims,
            'item_feat_dim': item_feat_dim,
            'best_auc': best_auc,
            'best_epoch': best_epoch + 1
        }
        
        # 记录模型元数据
        mlflow.log_dict(model_meta, 'model_meta.json')

if __name__ == '__main__':
    main() 