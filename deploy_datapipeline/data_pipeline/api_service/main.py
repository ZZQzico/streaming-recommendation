from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os
import logging
import random
import json

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)

# Load parent module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from kafka_stream.producer import send_interest_profile
from api_service.behavior_lookup import get_user_histories, get_recent_history, user_behavior
from metrics import get_metrics, set_kafka_sent_count
from config.redis_config import get_redis_connection

app = FastAPI()

# Read simulation range from env
RANDOM_USER_MIN = int(os.getenv("RANDOM_USER_MIN", 30))
RANDOM_USER_MAX = int(os.getenv("RANDOM_USER_MAX", 60))

class UserTimestampRequest(BaseModel):
    user_id: str
    timestamp: int  # Unix timestamp

class RecommendationRecord(BaseModel):
    user_id: str
    timestamp: int
    recommendation: list[str]

@app.post("/push_interest/")
def push_interest(req: UserTimestampRequest):
    """
    Push single user's interest profile to Kafka with timestamp
    """
    history_items = get_recent_history(req.user_id, req.timestamp)
    if not history_items:
        logging.warning(f"[Single] No history: user={req.user_id}, ts={req.timestamp}")
        return {"status": "error", "message": "No history found"}
    
    send_interest_profile(req.user_id, history_items, req.timestamp)
    set_kafka_sent_count(1)
    return {"status": "success", "user_id": req.user_id, "history_items_count": len(history_items)}

@app.post("/send_kafka/")
def send_kafka(req: UserTimestampRequest):
    """
    Simulate multiple users' profiles being pushed to Kafka.
    """
    sent = 0
    primary_history = get_recent_history(req.user_id, req.timestamp)
    if primary_history:
        send_interest_profile(req.user_id, primary_history, req.timestamp)
        sent += 1
    else:
        logging.warning(f"[Multi] No primary user history: {req.user_id}")
        return {"status": "error", "message": "No history found"}

    candidate_users = list(user_behavior.keys())
    candidate_users.remove(req.user_id)
    random_users = random.sample(candidate_users, k=random.randint(RANDOM_USER_MIN, RANDOM_USER_MAX))

    for user in random_users:
        other_history = get_recent_history(user, req.timestamp)
        if other_history:
            send_interest_profile(user, other_history, req.timestamp)
            sent += 1

    set_kafka_sent_count(sent)
    return {"message": f"Sent {sent} user profiles to Kafka"}

@app.post("/recommendation/")
def store_recommendation(record: RecommendationRecord):
    """
    Manually store a test recommendation result into Redis.
    """
    r = get_redis_connection()
    key = f"recommendation_result:{record.user_id}:{record.timestamp}"
    r.set(key, json.dumps(record.recommendation))
    return {"status": "success", "key": key}

@app.get("/recommendation/{user_id}/{timestamp}")
def get_recommendation(user_id: str, timestamp: int):
    """
    Query recommendation results from Redis.
    """
    r = get_redis_connection()
    key = f"recommendation_result:{user_id}:{timestamp}"
    val = r.get(key)
    return json.loads(val) if val else {"status": "error", "message": "No recommendation found"}

@app.get("/user_profile/{user_id}/{timestamp}")
def get_user_profile(user_id: str, timestamp: int):
    """
    Query user profile from Redis by user_id and timestamp.
    """
    r = get_redis_connection()
    key = f"user_profile:{user_id}:{timestamp}"
    val = r.get(key)
    return json.loads(val) if val else {"status": "error", "message": "No profile found"}

@app.get("/metrics/")
def metrics_view():
    return get_metrics()