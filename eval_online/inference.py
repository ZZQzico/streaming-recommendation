"""
api_service/inference.py
Fully implemented 3-stage recommendation pipeline with debug logging and forced numeric casting.
"""
import pandas as pd
import torch
import logging
from prometheus_client import Histogram
from .models import ITEM_EMB_PATH, ITEM_FEAT_DIM
from api_service.behavior_lookup import get_recent_history

# ------------- Debug Banner -------------
print("🚀 inference.py v3 loaded")

# ------------- Logging Setup -------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inference_debug")

# ---------------- Prometheus Timers ----------------
LC_TIMER = Histogram(
    "lightgcn_latency_s", "Latency for LightGCN recommendation stage"
)
DIN_TIMER = Histogram(
    "din_latency_s", "Latency for DIN reranking stage"
)
RN_TIMER = Histogram(
    "ranknet_latency_s", "Latency for RankNet final scoring stage"
)

# Load item embeddings once
df_items = pd.read_csv(ITEM_EMB_PATH, index_col=0)
# Ensure numeric
for col in df_items.columns:
    df_items[col] = df_items[col].astype(float)
_all_item_ids = df_items.index.tolist()

# ---------------- Helper Implementations ----------------
def prepare_din_inputs(user_id, item_ids, timestamp):
    logger.info(f"DEBUG prepare_din_inputs received item_ids first 3: {item_ids[:3]}")
    try:
        mat = df_items.loc[item_ids].values.astype(float)
    except Exception as e:
        logger.error(f"Failed df_items.loc with item_ids={item_ids[:5]}, error={e}")
        raise
    candidate_feats = torch.tensor(mat, dtype=torch.float32)
    # History
    hist_ids = get_recent_history(user_id, timestamp)
    max_seq = len(hist_ids)
    history_feats = torch.zeros((1, max_seq, ITEM_FEAT_DIM), dtype=torch.float32)
    for i, iid in enumerate(hist_ids):
        history_feats[0, i] = torch.tensor(
            df_items.loc[iid].values.astype(float),
            dtype=torch.float32
        )
    history_length = torch.tensor([max_seq], dtype=torch.int64)
    return candidate_feats, history_feats, history_length


def get_user_features(user_id):
    hist_ids = get_recent_history(user_id, timestamp=0)
    if not hist_ids:
        return torch.zeros((ITEM_FEAT_DIM,), dtype=torch.float32)
    mat = df_items.loc[hist_ids].values.astype(float)
    return torch.tensor(mat, dtype=torch.float32).mean(dim=0)


def get_item_features(item_id):
    vec = df_items.loc[item_id].values.astype(float)
    return torch.tensor(vec, dtype=torch.float32)


def rerank_top(din_scores, item_ids, top_n=20):
    pairs = list(zip(item_ids, din_scores.tolist()))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return [item for item, _ in pairs[:top_n]]

# ---------------- Recommendation Pipeline ----------------
def recommend_pipeline(user_id, timestamp, lightgcn_model, din_model, ranknet_model):
    print("🚀 recommend_pipeline called")
    # 1) Recall
    with LC_TIMER.time():
        scores = lightgcn_model.predict([user_id])
        positions = scores.topk(50).indices[0].tolist()
        topk_items = [_all_item_ids[pos] for pos in positions]
    logger.info(f"DEBUG Recall positions: {positions[:5]}")
    logger.info(f"DEBUG Recall item_ids:  {topk_items[:5]}")

    # 2) DIN
    with DIN_TIMER.time():
        cand_feats, hist_feats, hist_len = prepare_din_inputs(
            user_id, topk_items, timestamp
        )
        din_scores = din_model(cand_feats, hist_feats, hist_len)
        reranked = rerank_top(din_scores, topk_items)
    logger.info(f"DEBUG DIN reranked: {reranked[:5]}")

    # 3) RankNet
    with RN_TIMER.time():
        user_feat = get_user_features(user_id).unsqueeze(0)
        item_feat_list = [get_item_features(i).unsqueeze(0) for i in reranked]
        rn_scores = ranknet_model.get_rank_scores(user_feat, item_feat_list)
        final_pairs = list(zip(reranked, rn_scores[0].tolist()))
        final_pairs.sort(key=lambda x: x[1], reverse=True)
        final = [item for item, _ in final_pairs[:10]]
    logger.info(f"DEBUG Final: {final}")
    return final
