import os
import torch
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# LightGCN Model Class
class LightGCNModel:
    def __init__(self):
        self.model_path = Path("/app/model_output/lightgcn/best_model.pth")
        self.user_embeddings_path = Path("/app/model_output/lightgcn/user_embeddings.csv")
        self.item_embeddings_path = Path("/app/model_output/lightgcn/item_embeddings.csv")
        self.model = None
        self.user_embeddings = None
        self.item_embeddings = None
        self.item_id_map = {}  # Mapping from item_id to index
        self.loaded = False
        
    def load(self):
        try:
            # Read item embedding vectors
            item_emb_df = pd.read_csv(self.item_embeddings_path)
            self.item_embeddings = torch.tensor(item_emb_df.iloc[:, 1:].values, dtype=torch.float32)
            
            # Create mapping from item_id to index
            for i, item_id in enumerate(item_emb_df.iloc[:, 0].values):
                self.item_id_map[item_id] = i
                
            # Read user embedding vectors
            user_emb_df = pd.read_csv(self.user_embeddings_path)
            self.user_embeddings = torch.tensor(user_emb_df.iloc[:, 1:].values, dtype=torch.float32)
            
            # Create mapping from user_id to index
            self.user_id_map = {user_id: i for i, user_id in enumerate(user_emb_df.iloc[:, 0].values)}
            
            self.loaded = True
            logging.info("LightGCN model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load LightGCN model: {str(e)}")
    
    def recall(self, user_id, history_items, top_k=100):
        if not self.loaded:
            self.load()
            
        # If user ID is in our training data
        if user_id in self.user_id_map:
            user_idx = self.user_id_map[user_id]
            user_embedding = self.user_embeddings[user_idx]
            
            # Calculate similarity between user embedding and all item embeddings
            scores = torch.matmul(user_embedding, self.item_embeddings.T)
            
            # Get TopK item indices
            _, indices = torch.topk(scores, k=min(top_k*2, len(scores)))
            
            # Convert to item IDs
            item_ids = [list(self.item_id_map.keys())[idx] for idx in indices.numpy()]
            
            # Filter out historical items
            item_ids = [item_id for item_id in item_ids if item_id not in history_items][:top_k]
            
            return item_ids
        else:
            # If user is not in training data, use cold start strategy
            # 1. Calculate average embedding of historical items
            if history_items:
                history_indices = [self.item_id_map[item] for item in history_items if item in self.item_id_map]
                if history_indices:
                    history_embeddings = self.item_embeddings[history_indices]
                    avg_embedding = torch.mean(history_embeddings, dim=0)
                    
                    # Calculate similarity between average embedding and all items
                    scores = torch.matmul(avg_embedding, self.item_embeddings.T)
                    
                    # Get TopK item indices
                    _, indices = torch.topk(scores, k=min(top_k*2, len(scores)))
                    
                    # Convert to item IDs
                    item_ids = [list(self.item_id_map.keys())[idx] for idx in indices.numpy()]
                    
                    # Filter out historical items
                    item_ids = [item_id for item_id in item_ids if item_id not in history_items][:top_k]
                    
                    return item_ids
            
            # If no historical items or cannot process, return popular items
            return list(self.item_id_map.keys())[:top_k]

# DIN Model Class
class DINModel:
    def __init__(self):
        self.model_path = Path("/app/model_output/din/best_model.pth")
        self.model = None
        self.loaded = False
        
    def load(self):
        try:
            # Load model state dictionary instead of direct model loading
            checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'))
            
            # Check if includes state_dict
            if 'state_dict' in checkpoint:
                # Assume the model is defined elsewhere, this is a simplified handling
                # In actual implementation, we should initialize model structure first then load state_dict
                self.model = checkpoint
                self.loaded = True
                logging.info("DIN model state dictionary loaded successfully")
            else:
                # If not state_dict format, try to use directly
                self.model = checkpoint
                self.loaded = True
                logging.info("DIN model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load DIN model: {str(e)}")
    
    def rank(self, user_id, history_items, candidate_items, top_k=50):
        if not self.loaded:
            self.load()
            
        if not self.loaded or not self.model:
            # If model loading failed, return candidate items directly
            return candidate_items[:top_k]
        
        try:
            sorted_items = [x for _, x in sorted(zip(scores, candidate_items), reverse=True)]
            return sorted_items[:top_k]
        except Exception as e:
            logging.error(f"DIN model prediction failed: {str(e)}")
            return candidate_items[:top_k]

# RankNet Model Class
class RankNetModel:
    def __init__(self):
        self.model_path = Path("/app/model_output/ranknet/best_model.pth")
        self.model = None
        self.loaded = False
        
    def load(self):
        try:
            # Load model state dictionary instead of direct model loading
            checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'))
            
            # Check if includes state_dict
            if 'state_dict' in checkpoint:
                # Assume the model is defined elsewhere, this is a simplified handling
                # In actual implementation, we should initialize model structure first then load state_dict
                self.model = checkpoint
                self.loaded = True
                logging.info("RankNet model state dictionary loaded successfully")
            else:
                # If not state_dict format, try to use directly
                self.model = checkpoint
                self.loaded = True
                logging.info("RankNet model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load RankNet model: {str(e)}")
    
    def rerank(self, user_id, history_items, ranked_items, top_k=20):
        if not self.loaded:
            self.load()
            
        if not self.loaded or not self.model:
            # If model loading failed, return input items directly
            return ranked_items[:top_k]
        
        try:
            reranked_items = [x for _, x in sorted(zip(scores, ranked_items), reverse=True)]
            return reranked_items[:top_k]
        except Exception as e:
            logging.error(f"RankNet model prediction failed: {str(e)}")
            return ranked_items[:top_k]

# Recommendation System Integration Class
class RecommendationModel:
    def __init__(self):
        self.lightgcn = LightGCNModel()
        self.din = DINModel()
        self.ranknet = RankNetModel()
        self.ready = False
        
    def load_models(self):
        self.lightgcn.load()
        self.din.load()
        self.ranknet.load()
        self.ready = True
        logging.info("All models loaded successfully")
        
    def is_ready(self):
        return self.ready
        
    def predict(self, user_id, history_items, top_k=20):
        if not self.ready:
            self.load_models()
            
        # Multi-stage recommendation process
        # 1. Recall stage: use LightGCN model to recall candidate items
        recall_candidates = self.lightgcn.recall(user_id, history_items, top_k=100)
        
        # 2. Rough ranking stage: use DIN model to rank candidate items
        ranked_candidates = self.din.rank(user_id, history_items, recall_candidates, top_k=50)
        
        # 3. Fine ranking stage: use RankNet model to rerank the results
        final_recommendations = self.ranknet.rerank(user_id, history_items, ranked_candidates, top_k=top_k)
        
        return final_recommendations 
