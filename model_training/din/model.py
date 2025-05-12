import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    Attention layer implementation
    Used to calculate relevance weights between user history items and candidate item
    """
    def __init__(self, embedding_dim, attention_size):
        super(Attention, self).__init__()
        self.embedding_dim = embedding_dim
        self.attention_size = attention_size
        
        # Attention network parameters
        self.w_query = nn.Linear(embedding_dim, attention_size, bias=False)
        self.w_key = nn.Linear(embedding_dim, attention_size, bias=False)
        self.w_value = nn.Linear(attention_size, 1, bias=False)
        
    def forward(self, queries, keys, keys_length):
        """
        Calculate attention weights and return weighted representation
        
        Parameters:
            queries: Candidate item embedding vectors [batch_size, embedding_dim]
            keys: Historical item embedding vectors sequence [batch_size, max_seq_len, embedding_dim]
            keys_length: Actual length of history sequence for each user [batch_size]
        
        Returns:
            output: Attention-weighted user interest representation [batch_size, embedding_dim]
            attention_weights: Attention weights [batch_size, max_seq_len]
        """
        batch_size, max_seq_len = keys.size(0), keys.size(1)
        
        # Expand queries to match keys dimensions
        queries = queries.unsqueeze(1).expand(-1, max_seq_len, -1)  # [batch_size, max_seq_len, embedding_dim]
        
        # Calculate attention scores
        queries_hidden = self.w_query(queries)  # [batch_size, max_seq_len, attention_size]
        keys_hidden = self.w_key(keys)  # [batch_size, max_seq_len, attention_size]
        
        # Activation
        hidden = F.relu(queries_hidden + keys_hidden)  # [batch_size, max_seq_len, attention_size]
        
        # Calculate attention weights
        attention_scores = self.w_value(hidden).squeeze(-1)  # [batch_size, max_seq_len]
        
        # Create mask, set invalid history item positions to a very small value
        mask = torch.arange(max_seq_len, device=keys_length.device).unsqueeze(0).expand(batch_size, -1)
        mask = mask < keys_length.unsqueeze(-1)
        attention_scores = attention_scores.masked_fill(~mask, -1e9)
        
        # Apply softmax to get weights
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, max_seq_len]
        
        # Use weights to calculate weighted sum of history item embedding vectors
        output = torch.bmm(attention_weights.unsqueeze(1), keys).squeeze(1)  # [batch_size, embedding_dim]
        
        return output, attention_weights

class DIN(nn.Module):
    """
    Deep Interest Network model implementation
    Using attention mechanism to capture user's dynamic interest
    """
    def __init__(self, item_feat_dim, embedding_dim=64, attention_dim=64, mlp_hidden_dims=[128, 64, 32], dropout=0.2):
        super(DIN, self).__init__()
        self.embedding_dim = embedding_dim
        
        # Item embedding layer
        self.item_embedding = nn.Linear(item_feat_dim, embedding_dim)
        
        # Attention layer
        self.attention = Attention(embedding_dim, attention_dim)
        
        # MLP layers
        mlp_layers = []
        input_dim = embedding_dim * 3  # User interest vector + Candidate item vector + History item sequence average
        
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            mlp_layers.append(nn.BatchNorm1d(hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        self.mlp = nn.Sequential(*mlp_layers)
        self.output_layer = nn.Linear(mlp_hidden_dims[-1], 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, candidate_features, history_features, history_length):
        """
        Forward propagation function
        
        Parameters:
            candidate_features: Candidate item features [batch_size, item_feat_dim]
            history_features: Historical item feature sequence [batch_size, max_seq_len, item_feat_dim]
            history_length: Actual history sequence length for each user [batch_size]
        
        Returns:
            logits: Prediction scores [batch_size, 1]
        """
        batch_size, max_seq_len = history_features.size(0), history_features.size(1)
        
        # Convert raw features to embedding vectors
        candidate_emb = self.item_embedding(candidate_features)  # [batch_size, embedding_dim]
        
        # Process history item features
        history_emb = self.item_embedding(history_features.view(-1, history_features.size(-1)))
        history_emb = history_emb.view(batch_size, max_seq_len, -1)  # [batch_size, max_seq_len, embedding_dim]
        
        # Calculate attention-weighted user interest representation
        interest_emb, attention_weights = self.attention(candidate_emb, history_emb, history_length)
        
        # Calculate average vector of history items as additional feature
        mask = torch.arange(max_seq_len, device=history_length.device).unsqueeze(0) < history_length.unsqueeze(1)
        mask = mask.unsqueeze(-1).expand(-1, -1, self.embedding_dim).float()
        avg_history_emb = (history_emb * mask).sum(dim=1) / history_length.unsqueeze(-1).float()
        
        # Concatenate features
        concat_features = torch.cat([interest_emb, candidate_emb, avg_history_emb], dim=1)
        
        # Through MLP layers
        mlp_output = self.mlp(concat_features)
        
        # Output layer
        logits = self.output_layer(mlp_output)
        
        return self.sigmoid(logits).squeeze(-1)
    
    def calculate_loss(self, candidate_features, history_features, history_length, labels):
        """
        Calculate binary cross entropy loss
        
        Parameters:
            candidate_features: Candidate item features [batch_size, item_feat_dim]
            history_features: Historical item feature sequence [batch_size, max_seq_len, item_feat_dim]
            history_length: Actual history sequence length for each user [batch_size]
            labels: Ground truth labels [batch_size]
            
        Returns:
            loss: Cross entropy loss
        """
        pred = self.forward(candidate_features, history_features, history_length)
        loss = F.binary_cross_entropy(pred, labels.float())
        return loss 