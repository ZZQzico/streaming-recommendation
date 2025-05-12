import os
import torch
import pandas as pd
from .lightgcn import LightGCN
from .din      import DIN
from .ranknet  import RankNet

# --- Hyperparameters (hard-coded from inspection) ---
LGCN_EMBEDDING_DIM = 128
DIN_EMBED_DIM      = 128
DIN_ATTN_DIM       = 128
DIN_MLP_DIMS       = [128, 64, 32]
DIN_DROPOUT        = 0.2
# RankNet uses separate user/item feature dims
RN_EMBED_DIM       = 128
RN_USER_FEAT_DIM   = 128
RN_ITEM_FEAT_DIM   = 3
RN_HIDDEN_DIMS     = [128, 64, 32]
RN_DROPOUT         = 0.2

# --- File paths (override via env vars if needed) ---
LGCN_MODEL_PATH = os.getenv("LGCN_MODEL_PATH", "model/lightgcn.pth")
DIN_MODEL_PATH  = os.getenv("DIN_MODEL_PATH",  "model/din.pth")
RN_MODEL_PATH   = os.getenv("RN_MODEL_PATH",   "model/ranknet.pth")
ITEM_EMB_PATH   = os.getenv("ITEM_EMB_PATH",   "model/item_embeddings.csv")

# --- Load item features once (for DIN input) ---
_item_emb_df = pd.read_csv(ITEM_EMB_PATH, index_col=0)
ITEM_FEAT_DIM = _item_emb_df.shape[1]  # should be 3


def _load_state_dict(path, map_location=None):
    """
    Load checkpoint, unwrap common wrappers, strip 'module.' prefixes.
    """
    ckpt = torch.load(path, map_location=map_location)
    if isinstance(ckpt, dict):
        for key in ('state_dict', 'model_state_dict', 'state'):
            if key in ckpt:
                ckpt = ckpt[key]
                break
    return {k.replace('module.', ''): v for k, v in ckpt.items()}


def load_models(device: torch.device):
    """
    Load all three models for online inference.
    """
    # LightGCN
    lc = LightGCN(
        num_users     = _load_state_dict(LGCN_MODEL_PATH, 'cpu')['user_embedding.weight'].shape[0],
        num_items     = _load_state_dict(LGCN_MODEL_PATH, 'cpu')['item_embedding.weight'].shape[0],
        embedding_dim = LGCN_EMBEDDING_DIM
    )
    lc.load_state_dict(_load_state_dict(LGCN_MODEL_PATH, map_location='cpu'))
    lc.to(device).eval()

    # DIN
    din = DIN(
        item_feat_dim   = ITEM_FEAT_DIM,
        embedding_dim   = DIN_EMBED_DIM,
        attention_dim   = DIN_ATTN_DIM,
        mlp_hidden_dims = DIN_MLP_DIMS,
        dropout         = DIN_DROPOUT
    )
    din.load_state_dict(_load_state_dict(DIN_MODEL_PATH, map_location=device))
    din.to(device).eval()

    # RankNet
    rn = RankNet(
        user_feat_dim = RN_USER_FEAT_DIM,
        item_feat_dim = RN_ITEM_FEAT_DIM,
        embedding_dim = RN_EMBED_DIM,
        hidden_dims   = RN_HIDDEN_DIMS,
        dropout       = RN_DROPOUT
    )
    rn.load_state_dict(_load_state_dict(RN_MODEL_PATH, map_location=device))
    rn.to(device).eval()

    return lc, din, rn
