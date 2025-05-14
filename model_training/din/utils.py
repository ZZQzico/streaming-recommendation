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
    DIN model dataset
    Processes user history item sequences and candidate items
    """
    def __init__(self, interactions, item_features, max_seq_len=50):
        """
        Initialize dataset
        
        Parameters:
            interactions: Interaction data, including user history and candidate items
            item_features: Item feature data
            max_seq_len: Maximum length of history sequence
        """
        self.interactions = interactions
        self.item_features = item_features
        self.max_seq_len = max_seq_len
        
        # Item feature dimension
        self.item_feat_dim = len(self.item_features.columns) - 1  # Remove item_id column
        
        # Create item ID to feature mapping
        self.item_features_dict = self._create_item_features_dict()
        
    def _create_item_features_dict(self):
        """Create mapping from item ID to feature vector"""
        item_features_dict = {}
        
        for _, row in self.item_features.iterrows():
            item_id = row['item_id']
            features = row.drop('item_id').values.astype(np.float32)
            item_features_dict[item_id] = features
            
        # Create a default feature vector for unknown items
        default_features = np.zeros(self.item_feat_dim, dtype=np.float32)
        item_features_dict['UNK'] = default_features
        
        return item_features_dict
    
    def __len__(self):
        """Return dataset size"""
        return len(self.interactions)
    
    def __getitem__(self, idx):
        """Get a single sample"""
        interaction = self.interactions.iloc[idx]
        
        # Extract user ID, candidate item ID and label
        user_id = interaction['user_id']
        candidate_item = interaction['candidate_item']
        label = interaction['label']
        
        # Get history item list
        if pd.notna(interaction['history_items']) and interaction['history_items']:
            history_items = interaction['history_items'].split('|')
        else:
            history_items = []
        
        # Truncate or pad history sequence to max_seq_len
        if len(history_items) > self.max_seq_len:
            # Keep the most recent max_seq_len items
            history_items = history_items[-self.max_seq_len:]
        
        # Actual length of history sequence
        history_length = len(history_items)
        
        # Pad history sequence to max_seq_len
        history_items = history_items + ['UNK'] * (self.max_seq_len - len(history_items))
        
        # Get candidate item features
        candidate_features = self._get_item_features(candidate_item)
        
        # Get history item features
        history_features = np.array([self._get_item_features(item) for item in history_items])
        
        return {
            'user_id': user_id,
            'candidate_features': candidate_features,
            'history_features': history_features,
            'history_length': history_length,
            'label': label
        }
    
    def _get_item_features(self, item_id):
        """Get item feature vector"""
        if item_id in self.item_features_dict:
            return self.item_features_dict[item_id]
        else:
            return self.item_features_dict['UNK']

def load_data(train_data_path, item_embeddings_path, test_size=0.2, random_state=42, max_samples=None):
    """
    Load and process DIN model training data
    
    Parameters:
        train_data_path: Training data path
        item_embeddings_path: Item feature data path
        test_size: Validation set ratio
        random_state: Random seed
        max_samples: Maximum sample count, for debugging
        
    Returns:
        train_dataset: Training dataset
        valid_dataset: Validation dataset
        item_feat_dim: Item feature dimension
    """
    print(f"Loading training data: {train_data_path}")
    
    # Load interaction data
    interactions = pd.read_csv(train_data_path)
    
    # Limit sample count (for debugging)
    if max_samples and max_samples > 0:
        interactions = interactions.sample(min(max_samples, len(interactions)), random_state=random_state)
    
    print(f"Interaction data size: {len(interactions)}")
    
    # Extract all item IDs involved in interaction data
    print("Extracting required item IDs...")
    candidate_items = set(interactions['candidate_item'].unique())
    
    # Process history item lists
    history_items = set()
    for hist in tqdm(interactions['history_items'].dropna(), desc="Processing history items"):
        if isinstance(hist, str) and hist:
            items = hist.split('|')
            history_items.update(items)
    
    # Merge all needed item IDs
    needed_items = candidate_items.union(history_items)
    print(f"Number of item features to load: {len(needed_items)}")
    
    # Load item features
    print(f"Loading item features: {item_embeddings_path}")
    
    # Use chunked reading and filtering to avoid loading all data at once
    chunk_size = 500000  # Number of rows to read each time
    filtered_items = []
    
    print("Reading item features in chunks...")
    for chunk in tqdm(pd.read_csv(item_embeddings_path, chunksize=chunk_size), desc="Filtering item features"):
        # Only keep needed items
        filtered_chunk = chunk[chunk['item_id'].isin(needed_items)]
        filtered_items.append(filtered_chunk)
        
        # If all needed items have been found, exit early
        if len(filtered_items) > 0 and len(pd.concat(filtered_items)['item_id'].unique()) >= len(needed_items):
            break
    
    # Merge all filtered data chunks
    item_features = pd.concat(filtered_items, ignore_index=True)
    
    # Add UNK item
    if 'UNK' not in item_features['item_id'].values:
        # Create UNK item features (all zeros)
        unk_features = pd.DataFrame({
            'item_id': ['UNK'],
            'category_hash': [0.0],
            'brand_hash': [0.0],
            'price_scaled': [0.0]
        })
        item_features = pd.concat([item_features, unk_features], ignore_index=True)
    
    print(f"Filtered item feature data size: {len(item_features)}")
    
    # Item feature dimension
    item_feat_dim = len(item_features.columns) - 1  # Remove item_id column
    
    # Split training and validation sets
    train_interactions, valid_interactions = train_test_split(
        interactions, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set size: {len(train_interactions)}, Validation set size: {len(valid_interactions)}")
    
    # Create datasets
    train_dataset = DINDataset(train_interactions, item_features)
    valid_dataset = DINDataset(valid_interactions, item_features)
    
    return train_dataset, valid_dataset, item_feat_dim

def collate_fn(batch):
    """
    Data batch processing function
    
    Parameters:
        batch: Batch data
        
    Returns:
        Processed batch data
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
    Create data loaders
    
    Parameters:
        train_dataset: Training dataset
        valid_dataset: Validation dataset
        batch_size: Batch size
        num_workers: Number of worker threads
        
    Returns:
        train_loader: Training data loader
        valid_loader: Validation data loader
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
    Evaluate model performance
    
    Parameters:
        model: DIN model
        data_loader: Data loader
        device: Computation device
        
    Returns:
        metrics: Evaluation metrics dictionary
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
            
            # Forward propagation
            preds = model(candidate_features, history_features, history_lengths)
            
            # Calculate loss
            loss = model.calculate_loss(candidate_features, history_features, history_lengths, labels)
            total_loss += loss.item() * len(labels)
            
            # Collect predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate evaluation metrics
    from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Binary classification threshold
    binary_preds = (all_preds >= 0.5).astype(int)
    
    # Calculate AUC
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except:
        auc = 0.5  # If all labels are the same class, AUC calculation will error
    
    # Calculate Log Loss
    logloss = log_loss(all_labels, all_preds)
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, binary_preds)
    
    # Calculate average loss
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
    Create or get MLflow experiment
    
    Parameters:
        experiment_name: Experiment name
        tracking_uri: MLflow tracking server address
        
    Returns:
        experiment_id: Experiment ID
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    # Get or create experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(experiment_name)
    
    return experiment_id

def save_model(model, save_path):
    """
    Save model and its configuration
    
    Parameters:
        model: Model object
        save_path: Save path
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save model parameters
    torch.save({
        'state_dict': model.state_dict(),
        'config': {
            'embedding_dim': model.embedding_dim,
            'attention_dim': model.attention.attention_size,
            'mlp_hidden_dims': [layer.out_features for layer in model.mlp if isinstance(layer, nn.Linear)],
        }
    }, save_path)
    
    print(f"Model saved to {save_path}")

def load_model(model_path, item_feat_dim, device):
    """
    Load saved model
    
    Parameters:
        model_path: Model file path
        item_feat_dim: Item feature dimension
        device: Computation device
        
    Returns:
        model: Loaded model
    """
    from model import DIN
    
    # Load model parameters
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model configuration
    config = checkpoint['config']
    
    # Create model
    model = DIN(
        item_feat_dim=item_feat_dim,
        embedding_dim=config['embedding_dim'],
        attention_dim=config['attention_dim'],
        mlp_hidden_dims=config['mlp_hidden_dims']
    ).to(device)
    
    # Load model parameters
    model.load_state_dict(checkpoint['state_dict'])
    
    return model 