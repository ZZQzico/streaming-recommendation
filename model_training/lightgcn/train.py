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
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size')
    parser.add_argument('--early_stop', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--n_neg', type=int, default=1, help='Number of negative samples per positive sample')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of processes for sampling and data loading, 0 means automatic selection')
    
    # MLflow parameters
    parser.add_argument('--experiment_name', type=str, default='LightGCN', help='MLflow experiment name')
    parser.add_argument('--tracking_uri', type=str, default=None, help='MLflow tracking server address')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Computation device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_embeddings', action='store_true', help='Whether to save embeddings')
    
    return parser.parse_args()

def train(model, optimizer, train_loader, device, reg_weight=1e-5):
    """Train one epoch"""
    model.train()
    total_loss = 0.0
    total_reg_loss = 0.0
    total_samples = 0
    
    for batch in tqdm(train_loader, desc='Training'):
        user, pos_item, neg_item = batch
        user = user.to(device)
        pos_item = pos_item.to(device)
        neg_item = neg_item.to(device)
        
        # Calculate BPR loss
        loss, reg_loss = model.bpr_loss(user, pos_item, neg_item)
        
        # Total loss = BPR loss + Regularization loss
        batch_loss = loss + reg_weight * reg_loss
        
        # Backpropagation
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        
        # Accumulate statistics
        batch_size = user.size(0)
        total_loss += loss.item() * batch_size
        total_reg_loss += reg_loss.item() * batch_size
        total_samples += batch_size
    
    # Calculate average loss
    avg_loss = float(total_loss / total_samples)
    avg_reg_loss = float(total_reg_loss / total_samples)
    
    return avg_loss, avg_reg_loss

def save_embeddings_fixed(model, user_mapping, item_mapping, save_dir):
    """
    Save trained user and item embeddings (fixed version)
    
    Parameters:
        model: Trained LightGCN model
        user_mapping: Mapping from user ID to index
        item_mapping: Mapping from item ID to index
        save_dir: Save path
    """
    model.eval()
    
    # Get final user and item embeddings
    with torch.no_grad():
        device = next(model.parameters()).device
        edge_index = model.edge_index.to(device)
        user_emb, item_emb = model.forward(edge_index)
    
    # Create reverse mapping
    idx_to_user = {idx: user for user, idx in user_mapping.items()}
    idx_to_item = {idx: item for item, idx in item_mapping.items()}
    
    # Save user embeddings
    user_emb_df = pd.DataFrame()
    user_emb_df['user_id'] = [idx_to_user[i] for i in range(len(user_mapping))]
    
    for i in range(user_emb.shape[1]):
        user_emb_df[f'emb_{i}'] = user_emb[:, i].cpu().numpy()
    
    # Save item embeddings
    item_emb_df = pd.DataFrame()
    item_emb_df['item_id'] = [idx_to_item[i] for i in range(len(item_mapping))]
    
    for i in range(item_emb.shape[1]):
        item_emb_df[f'emb_{i}'] = item_emb[:, i].cpu().numpy()
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Save as CSV files
    user_emb_df.to_csv(os.path.join(save_dir, 'user_embeddings.csv'), index=False)
    item_emb_df.to_csv(os.path.join(save_dir, 'item_embeddings.csv'), index=False)
    
    print(f"Embeddings saved to {save_dir}")

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create MLflow experiment
    experiment_id = create_mlflow_experiment(args.experiment_name, args.tracking_uri)
    
    # Determine maximum number of users and items
    max_users = args.max_users if args.max_users > 0 else None
    max_items = args.max_items if args.max_items > 0 else None
    
    # Load data
    train_data, valid_data, user_mapping, item_mapping, n_users, n_items = load_data(
        args.data_path, 
        test_size=args.test_size, 
        random_state=args.seed,
        max_users=max_users,
        max_items=max_items
    )
    
    # Create user-item bipartite graph
    graph_data = create_user_item_graph(train_data, n_users, n_items)
    edge_index = graph_data.edge_index
    
    # Create model
    model = LightGCN(
        num_users=n_users,
        num_items=n_items,
        embedding_dim=args.embedding_dim,
        num_layers=args.n_layers,
        dropout=args.dropout
    )
    
    # Set model edge index
    model.set_edge_index(edge_index)
    
    # Move model to specified device
    model = model.to(args.device)
    edge_index = edge_index.to(args.device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Determine number of worker processes
    num_workers = args.num_workers if args.num_workers > 0 else None
    dataloader_workers = max(4, args.num_workers) if args.num_workers > 0 else 4
    
    # Generate training samples (positive-negative pairs)
    print("Generating training samples...")
    user_indices, pos_item_indices, neg_item_indices = sample_negative_items(
        train_data, n_items, n_neg=args.n_neg, num_workers=num_workers
    )
    
    print(f"Number of generated training samples: {len(user_indices)}")
    
    # Create data loader
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
    
    # Generate candidate items for evaluation
    print("Generating candidate items...")
    valid_users = valid_data['user_idx'].unique()
    all_item_indices = np.arange(n_items)
    user_candidates, user_positives = generate_candidates(
        valid_users, valid_data, all_item_indices, n_candidates=100
    )
    
    # Training loop
    best_recall = 0
    best_epoch = 0
    early_stop_counter = 0
    
    with mlflow.start_run(experiment_id=experiment_id) as run:
        # Record parameters
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
        
        print("Starting training...")
        for epoch in range(args.n_epochs):
            # Train one epoch
            start_time = time.time()
            train_loss, reg_loss = train(model, optimizer, train_loader, args.device)
            train_time = time.time() - start_time
            
            # Evaluate
            eval_users = list(user_candidates.keys())
            metrics = calculate_metrics(model, eval_users, user_candidates, user_positives, args.device)
            
            # Print results
            print(f"Epoch {epoch+1}/{args.n_epochs} - "
                  f"Training loss: {train_loss:.4f}, Regularization loss: {reg_loss:.4f}, "
                  f"Time: {train_time:.2f}s, "
                  f"Recall@10: {metrics['recall_at_10']:.4f}, NDCG@10: {metrics['ndcg_at_10']:.4f}")
            
            # Ensure all metrics are scalar floats
            log_metrics = {
                'train_loss': float(train_loss),
                'reg_loss': float(reg_loss),
                'train_time': float(train_time)
            }
            
            # Add evaluation metrics
            for k, v in metrics.items():
                log_metrics[k] = float(v)
            
            # Record metrics
            mlflow.log_metrics(log_metrics, step=epoch)
            
            # Check if it's the best model
            if metrics['recall_at_10'] > best_recall:
                best_recall = metrics['recall_at_10']
                best_epoch = epoch
                early_stop_counter = 0
                
                # Save best model
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
                
                # Record best model
                mlflow.pytorch.log_model(model, "best_model")
            else:
                early_stop_counter += 1
                
            # Early stopping
            if early_stop_counter >= args.early_stop:
                print(f"Early stopping: No improvement for {args.early_stop} epochs")
                break
        
        # Load best model
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pth')))
        
        # Record best metrics
        final_metrics = calculate_metrics(model, eval_users, user_candidates, user_positives, args.device)
        print(f"Best model (Epoch {best_epoch+1}):")
        for k, v in final_metrics.items():
            print(f"  {k}: {v:.4f}")
        
        # Ensure all metrics are scalar floats
        best_metrics = {}
        for k, v in final_metrics.items():
            best_metrics[f'best_{k}'] = float(v)
        
        mlflow.log_metrics(best_metrics)
        
        # Save embeddings
        if args.save_embeddings:
            save_embeddings_fixed(model, user_mapping, item_mapping, args.output_dir)
    
    print("Training completed!")

if __name__ == '__main__':
    main() 