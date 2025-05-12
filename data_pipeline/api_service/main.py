from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import sys
import os
import logging
import random

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # Load parent module
from kafka_stream.producer import send_interest_profile
from api_service.behavior_lookup import get_user_histories, get_recent_history, user_behavior
from metrics import get_metrics, set_kafka_sent_count

app = FastAPI()

# Read min and max number of other users from .env
RANDOM_USER_MIN = int(os.getenv("RANDOM_USER_MIN", 30))
RANDOM_USER_MAX = int(os.getenv("RANDOM_USER_MAX", 60))

class UserTimestampRequest(BaseModel):
    user_id: str
    timestamp: int  # Unix timestamp

class TimestampRequest(BaseModel):
    timestamp: int

# === Single user request endpoint ===
@app.post("/push_interest/")
def push_interest(req: UserTimestampRequest):
    history_items = get_recent_history(req.user_id, req.timestamp)

    if not history_items:
        logging.warning(f"[Single] No history behavior: user={req.user_id}, ts={req.timestamp}")
        return {"status": "error", "message": "No history found"}

    send_interest_profile(req.user_id, history_items)
    set_kafka_sent_count(1)
    logging.info(f"[Single] Sent user={req.user_id}, history_len={len(history_items)}")
    return {"status": "success", "user_id": req.user_id, "history_items_count": len(history_items)}

# === Multi-user batch simulation request ===
@app.post("/send_kafka/")
def send_kafka(req: UserTimestampRequest):
    sent = 0

    # 1. Main user
    history_items = get_recent_history(req.user_id, req.timestamp)
    if history_items:
        send_interest_profile(req.user_id, history_items)
        sent += 1
    else:
        logging.warning(f"[Multi] No main user behavior: user={req.user_id}, ts={req.timestamp}")
        return {"status": "error", "message": "No history found"}

    # 2. Randomly simulate other users
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

    logging.info(f"[Multi] Main user + {len(random_users)} other users, sent {sent} user profiles, timestamp={req.timestamp}")
    return {"message": f"Sent {sent} user profiles to Kafka"}

# === Monitoring endpoint for Prometheus/Grafana ===
@app.get("/metrics/")
def metrics_view():
    return get_metrics()