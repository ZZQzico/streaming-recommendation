from fastapi import FastAPI, Response
from pydantic import BaseModel
import sys
import os
import logging
import random
import torch

# 将项目根目录加入路径，以便导入 api_service 子模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 行为生产与指标
from kafka_stream.producer import send_interest_profile
from api_service.behavior_lookup import get_user_histories, get_recent_history, user_behavior
from api_service.metrics import get_metrics, set_kafka_sent_count, req_errors

# 模型加载与推理
from api_service.models import load_models
from api_service.inference import recommend_pipeline

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)

app = FastAPI()

# 环境变量配置
RANDOM_USER_MIN = int(os.getenv("RANDOM_USER_MIN", 30))
RANDOM_USER_MAX = int(os.getenv("RANDOM_USER_MAX", 60))

# ---------- 启动时加载模型 ----------
@app.on_event("startup")
def startup_event():
    global DEVICE, LC_MODEL, DIN_MODEL, RN_MODEL
    DEVICE = torch.device(os.getenv("DEVICE", "cpu"))
    LC_MODEL, DIN_MODEL, RN_MODEL = load_models(DEVICE)
    logging.info("[Startup] Models loaded: LightGCN, DIN, RankNet")


# 请求体定义
class UserTimestampRequest(BaseModel):
    user_id: str
    timestamp: int  # Unix 时间戳

class PredictRequest(BaseModel):
    user_id: str
    timestamp: int


# === 单用户请求入口 ===
@app.post("/push_interest/")
def push_interest(req: UserTimestampRequest):
    history_items = get_recent_history(req.user_id, req.timestamp)
    if not history_items:
        logging.warning(f"[Single] 无历史行为: user={req.user_id}, ts={req.timestamp}")
        return {"status": "error", "message": "No history found"}

    send_interest_profile(req.user_id, history_items)
    set_kafka_sent_count(1)
    logging.info(f"[Single] 已发送 user={req.user_id}, history_len={len(history_items)}")
    return {"status": "success", "user_id": req.user_id, "history_items_count": len(history_items)}


# === 多用户批量模拟请求 ===
@app.post("/send_kafka/")
def send_kafka(req: UserTimestampRequest):
    sent = 0
    # 主用户
    history_items = get_recent_history(req.user_id, req.timestamp)
    if history_items:
        send_interest_profile(req.user_id, history_items)
        sent += 1
    else:
        logging.warning(f"[Multi] 无主用户行为: user={req.user_id}, ts={req.timestamp}")
        return {"status": "error", "message": "No history found"}

    # 随机模拟其他用户
    candidate_users = list(user_behavior.keys())
    if req.user_id in candidate_users:
        candidate_users.remove(req.user_id)
    num_random = random.randint(RANDOM_USER_MIN, RANDOM_USER_MAX)
    random_users = random.sample(candidate_users, k=num_random)
    for other_user in random_users:
        other_history = get_recent_history(other_user, req.timestamp)
        if other_history:
            send_interest_profile(other_user, other_history)
            sent += 1

    set_kafka_sent_count(sent)
    logging.info(f"[Multi] 主用户 + {len(random_users)} 个其他用户，共发送 {sent} 条用户画像，timestamp={req.timestamp}")
    return {"message": f"已发送 {sent} 条用户画像到 Kafka"}


# === 三阶段模型推理接口 ===
@app.post("/predict/")
def predict(req: PredictRequest):
    try:
        recs = recommend_pipeline(
            req.user_id, req.timestamp,
            LC_MODEL, DIN_MODEL, RN_MODEL
        )
        return {"status": "success", "recommendations": recs}
    except Exception as e:
        req_errors.labels(endpoint="/predict/").inc()
        logging.error(f"[Predict] Error for user={req.user_id}, error={e}")
        return {"status": "error", "message": str(e)}


# === Prometheus/Grafana 可调用的监控端点 ===
@app.get("/metrics/")
def metrics_view():
    return get_metrics()
