# api_service/lightgcn.py

import torch
import torch.nn as nn

class LightGCNConv(nn.Module):
    """No-op stub: identity pass-through."""
    def __init__(self, **kwargs):
        super().__init__()
    def forward(self, x, edge_index=None):
        return x

class LightGCN(nn.Module):
    """
    Simplified LightGCN for inference:
      - only user & item embeddings
      - predict() = inner product of embeddings
    """
    def __init__(self,
                 num_users: int,
                 num_items: int,
                 embedding_dim: int = 64,
                 num_layers: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        # same init as before
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def predict(self, users):
        """
        users: either a list of user indices or a LongTensor
        returns: Tensor of shape [len(users), num_items]
        """
        if not torch.is_tensor(users):
            users = torch.LongTensor(users).to(self.user_embedding.weight.device)
        # (batch, dim)
        user_emb = self.user_embedding(users)
        # (num_items, dim)
        item_emb = self.item_embedding.weight
        # (batch, num_items)
        scores = torch.matmul(user_emb, item_emb.t())
        return scores
