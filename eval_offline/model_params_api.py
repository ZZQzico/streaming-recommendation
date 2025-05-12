from fastapi import FastAPI, HTTPException
from typing import Dict, Any
import torch

# 假设模型类定义在 models 包中，需要根据实际路径调整
from models.lightgcn import LightGCN
from models.din import DINModel
from models.ranknet import RankNetModel

app = FastAPI()

# 模型文件路径配置
MODEL_PATHS = {
    "lightgcn": "/path/to/lightgcn.pth",
    "din": "/path/to/din.pth",
    "ranknet": "/path/to/ranknet.pth",
}

# 模型构造函数映射，需要根据模型初始化参数调整
MODEL_CLASSES = {
    "lightgcn": LightGCN,
    "din": DINModel,
    "ranknet": RankNetModel,
}

@app.get("/params/{model_name}")
def get_model_params(model_name: str) -> Dict[str, Any]:
    """
    根据模型名称加载对应模型并返回其参数字典。
    :param model_name: lightgcn | din | ranknet
    :return: 模型参数字典，键为参数名，值为列表形式的参数数据。
    """
    # 检查模型名称
    if model_name not in MODEL_PATHS:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")

    ckpt_path = MODEL_PATHS[model_name]
    ModelClass = MODEL_CLASSES[model_name]

    # 加载 checkpoint
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"Checkpoint file not found: {ckpt_path}")

    # 初始化模型实例，比如LightGCN需要传入节点数、层数等参数
    # 下面示例使用默认构造，无参数；根据实际情况修改
    model = ModelClass()

    # 加载权重
    try:
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load state_dict: {e}")

    # 提取并序列化参数
    params = {}
    for name, tensor in model.state_dict().items():
        params[name] = tensor.cpu().numpy().tolist()

    return {"model": model_name, "parameters": params}

# 示例启动命令: uvicorn model_params_api:app --host 0.0.0.0 --port 8000
