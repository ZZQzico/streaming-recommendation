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
    """Set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='DIN Training Parameters')
    
    # Data parameters
    parser.add_argument('--train_data_path', type=str, default='../../data_processing/train_data.csv', help='Training data path')
    parser.add_argument('--item_embeddings_path', type=str, default='../../data_processing/item_embeddings.csv', help='Item feature data path')
    parser.add_argument('--output_dir', type=str, default='../../model_output/din', help='Model output path')
    parser.add_argument('--max_samples', type=int, default=0, help='Maximum sample count for debugging, set to 0 to use all samples')
    
    # Model parameters
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--attention_dim', type=int, default=128, help='Attention layer hidden dimension')
    parser.add_argument('--mlp_hidden_dims', type=str, default='128,64,32', help='MLP hidden layer dimensions, comma separated')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--n_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--early_stop', type=int, default=100, help='Early stopping patience')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading worker threads')
    
    # MLflow parameters
    parser.add_argument('--experiment_name', type=str, default='DIN', help='MLflow experiment name')
    parser.add_argument('--tracking_uri', type=str, default=None, help='MLflow tracking server address')
    
    # Other parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Computation device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    return parser.parse_args()

def train_epoch(model, optimizer, train_loader, device):
    """Train one epoch"""
    model.train()
    total_loss = 0
    total_samples = 0
    
    for batch in tqdm(train_loader, desc='Training'):
        # Get data
        candidate_features = batch['candidate_features'].to(device)
        history_features = batch['history_features'].to(device)
        history_lengths = batch['history_lengths'].to(device)
        labels = batch['labels'].to(device)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Calculate loss
        loss = model.calculate_loss(candidate_features, history_features, history_lengths, labels)
        
        # Backpropagation
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Statistics
        batch_size = candidate_features.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    
    # Calculate average loss
    avg_loss = total_loss / total_samples
    
    return avg_loss

def validate(model, valid_loader, device):
    """Validate model"""
    metrics = evaluate_model(model, valid_loader, device)
    return metrics

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create MLflow experiment
    experiment_id = create_mlflow_experiment(args.experiment_name, args.tracking_uri)
    
    # Load data
    max_samples = args.max_samples if args.max_samples > 0 else None
    train_dataset, valid_dataset, item_feat_dim = load_data(
        args.train_data_path, 
        args.item_embeddings_path,
        max_samples=max_samples
    )
    
    # Create data loaders
    train_loader, valid_loader = create_data_loaders(
        train_dataset, 
        valid_dataset, 
        args.batch_size, 
        args.num_workers
    )
    
    # Parse MLP hidden layer dimensions
    mlp_hidden_dims = [int(dim) for dim in args.mlp_hidden_dims.split(',')]
    
    # Create model
    model = DIN(
        item_feat_dim=item_feat_dim,
        embedding_dim=args.embedding_dim,
        attention_dim=args.attention_dim,
        mlp_hidden_dims=mlp_hidden_dims,
        dropout=args.dropout
    ).to(args.device)
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=2, 
        verbose=True
    )
    
    # Training loop
    best_auc = 0
    best_epoch = 0
    early_stop_counter = 0
    
    with mlflow.start_run(experiment_id=experiment_id) as run:
        # Record parameters
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
            # Train
            train_loss = train_epoch(model, optimizer, train_loader, args.device)
            
            # Validate
            valid_metrics = validate(model, valid_loader, args.device)
            valid_loss = valid_metrics['loss']
            valid_auc = valid_metrics['auc']
            valid_logloss = valid_metrics['logloss']
            valid_accuracy = valid_metrics['accuracy']
            
            # Update learning rate
            scheduler.step(valid_loss)
            
            # Print metrics
            print(f"Epoch {epoch+1}/{args.n_epochs} - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Valid Loss: {valid_loss:.4f}, "
                  f"Valid AUC: {valid_auc:.4f}, "
                  f"Valid LogLoss: {valid_logloss:.4f}, "
                  f"Valid Accuracy: {valid_accuracy:.4f}")
            
            # Record metrics
            mlflow.log_metrics({
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'valid_auc': valid_auc,
                'valid_logloss': valid_logloss,
                'valid_accuracy': valid_accuracy
            }, step=epoch)
            
            # Check if it's the best model
            if valid_auc > best_auc:
                best_auc = valid_auc
                best_epoch = epoch
                early_stop_counter = 0
                
                # Save best model
                model_path = os.path.join(args.output_dir, 'best_model.pth')
                save_model(model, model_path)
                
                print(f"Saved best model, validation AUC: {valid_auc:.4f}")
            else:
                early_stop_counter += 1
                
            # Early stopping
            if early_stop_counter >= args.early_stop:
                print(f"Early stopping triggered, best model occurred at Epoch {best_epoch+1}")
                break
        
        # Record best performance
        mlflow.log_metrics({
            'best_auc': best_auc,
            'best_epoch': best_epoch + 1
        })
        
        print(f"Training completed, best validation AUC: {best_auc:.4f}, best Epoch: {best_epoch+1}")
        
        # Save embeddings and model configuration
        model_meta = {
            'model_type': 'DIN',
            'embedding_dim': args.embedding_dim,
            'attention_dim': args.attention_dim,
            'mlp_hidden_dims': mlp_hidden_dims,
            'item_feat_dim': item_feat_dim,
            'best_auc': best_auc,
            'best_epoch': best_epoch + 1
        }
        
        # Record model metadata
        mlflow.log_dict(model_meta, 'model_meta.json')

if __name__ == "__main__":
    main() 