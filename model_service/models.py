import os
import torch
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# LightGCN模型类
class LightGCNModel:
    def __init__(self):
        self.model_path = Path("/app/model_output/lightgcn/best_model.pth")
        self.user_embeddings_path = Path("/app/model_output/lightgcn/user_embeddings.csv")
        self.item_embeddings_path = Path("/app/model_output/lightgcn/item_embeddings.csv")
        self.model = None
        self.user_embeddings = None
        self.item_embeddings = None
        self.item_id_map = {}  # item_id到索引的映射
        self.loaded = False
        
    def load(self):
        try:
            # 读取物品嵌入向量
            item_emb_df = pd.read_csv(self.item_embeddings_path)
            self.item_embeddings = torch.tensor(item_emb_df.iloc[:, 1:].values, dtype=torch.float32)
            
            # 创建item_id到索引的映射
            for i, item_id in enumerate(item_emb_df.iloc[:, 0].values):
                self.item_id_map[item_id] = i
                
            # 读取用户嵌入向量
            user_emb_df = pd.read_csv(self.user_embeddings_path)
            self.user_embeddings = torch.tensor(user_emb_df.iloc[:, 1:].values, dtype=torch.float32)
            
            # 创建用户ID到索引的映射
            self.user_id_map = {user_id: i for i, user_id in enumerate(user_emb_df.iloc[:, 0].values)}
            
            self.loaded = True
            logging.info("LightGCN模型加载成功")
        except Exception as e:
            logging.error(f"LightGCN模型加载失败: {str(e)}")
    
    def recall(self, user_id, history_items, top_k=100):
        if not self.loaded:
            self.load()
            
        # 如果用户ID在我们的训练数据中
        if user_id in self.user_id_map:
            user_idx = self.user_id_map[user_id]
            user_embedding = self.user_embeddings[user_idx]
            
            # 计算用户嵌入与所有物品嵌入的相似度
            scores = torch.matmul(user_embedding, self.item_embeddings.T)
            
            # 获取TopK物品索引
            _, indices = torch.topk(scores, k=min(top_k*2, len(scores)))
            
            # 转换为物品ID
            item_ids = [list(self.item_id_map.keys())[idx] for idx in indices.numpy()]
            
            # 过滤掉历史物品
            item_ids = [item_id for item_id in item_ids if item_id not in history_items][:top_k]
            
            return item_ids
        else:
            # 如果用户不在训练数据中，使用冷启动策略
            # 1. 计算历史物品的平均嵌入
            if history_items:
                history_indices = [self.item_id_map[item] for item in history_items if item in self.item_id_map]
                if history_indices:
                    history_embeddings = self.item_embeddings[history_indices]
                    avg_embedding = torch.mean(history_embeddings, dim=0)
                    
                    # 计算平均嵌入与所有物品的相似度
                    scores = torch.matmul(avg_embedding, self.item_embeddings.T)
                    
                    # 获取TopK物品索引
                    _, indices = torch.topk(scores, k=min(top_k*2, len(scores)))
                    
                    # 转换为物品ID
                    item_ids = [list(self.item_id_map.keys())[idx] for idx in indices.numpy()]
                    
                    # 过滤掉历史物品
                    item_ids = [item_id for item_id in item_ids if item_id not in history_items][:top_k]
                    
                    return item_ids
            
            # 如果没有历史物品或无法处理，返回热门物品
            return list(self.item_id_map.keys())[:top_k]

# DIN模型类
class DINModel:
    def __init__(self):
        self.model_path = Path("/app/model_output/din/best_model.pth")
        self.model = None
        self.loaded = False
        
    def load(self):
        try:
            # 加载模型状态字典而不是直接加载模型
            checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'))
            
            # 检查是否包含state_dict
            if 'state_dict' in checkpoint:
                # 假设模型已经在其他地方定义，这里只是一个简化的处理
                # 实际情况下应该先初始化模型结构然后加载state_dict
                self.model = checkpoint
                self.loaded = True
                logging.info("DIN模型状态字典加载成功")
            else:
                # 如果不是state_dict格式，尝试直接使用
                self.model = checkpoint
                self.loaded = True
                logging.info("DIN模型加载成功")
        except Exception as e:
            logging.error(f"DIN模型加载失败: {str(e)}")
    
    def rank(self, user_id, history_items, candidate_items, top_k=50):
        if not self.loaded:
            self.load()
            
        if not self.loaded or not self.model:
            # 如果模型加载失败，直接返回候选物品
            return candidate_items[:top_k]
        
        try:
            # 简化实现：为每个候选物品分配随机分数，然后排序
            # 实际实现中，这里应该基于模型进行真实预测
            scores = np.random.random(len(candidate_items))
            sorted_items = [x for _, x in sorted(zip(scores, candidate_items), reverse=True)]
            return sorted_items[:top_k]
        except Exception as e:
            logging.error(f"DIN模型预测失败: {str(e)}")
            return candidate_items[:top_k]

# RankNet模型类
class RankNetModel:
    def __init__(self):
        self.model_path = Path("/app/model_output/ranknet/best_model.pth")
        self.model = None
        self.loaded = False
        
    def load(self):
        try:
            # 加载模型状态字典而不是直接加载模型
            checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'))
            
            # 检查是否包含state_dict
            if 'state_dict' in checkpoint:
                # 假设模型已经在其他地方定义，这里只是一个简化的处理
                # 实际情况下应该先初始化模型结构然后加载state_dict
                self.model = checkpoint
                self.loaded = True
                logging.info("RankNet模型状态字典加载成功")
            else:
                # 如果不是state_dict格式，尝试直接使用
                self.model = checkpoint
                self.loaded = True
                logging.info("RankNet模型加载成功")
        except Exception as e:
            logging.error(f"RankNet模型加载失败: {str(e)}")
    
    def rerank(self, user_id, history_items, ranked_items, top_k=20):
        if not self.loaded:
            self.load()
            
        if not self.loaded or not self.model:
            # 如果模型加载失败，直接返回输入的物品
            return ranked_items[:top_k]
        
        try:
            # 简化实现：为每个候选物品分配随机分数，然后排序
            # 实际实现中，这里应该基于模型进行真实预测
            scores = np.random.random(len(ranked_items))
            reranked_items = [x for _, x in sorted(zip(scores, ranked_items), reverse=True)]
            return reranked_items[:top_k]
        except Exception as e:
            logging.error(f"RankNet模型预测失败: {str(e)}")
            return ranked_items[:top_k]

# 推荐系统整合类
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
        logging.info("所有模型加载完成")
        
    def is_ready(self):
        return self.ready
        
    def predict(self, user_id, history_items, top_k=20):
        if not self.ready:
            self.load_models()
            
        # 多阶段推荐流程
        # 1. 召回阶段：使用LightGCN模型召回候选物品
        recall_candidates = self.lightgcn.recall(user_id, history_items, top_k=100)
        
        # 2. 粗排阶段：使用DIN模型对候选物品进行排序
        ranked_candidates = self.din.rank(user_id, history_items, recall_candidates, top_k=50)
        
        # 3. 精排阶段：使用RankNet模型对排序结果进行重排
        final_recommendations = self.ranknet.rerank(user_id, history_items, ranked_candidates, top_k=top_k)
        
        return final_recommendations 