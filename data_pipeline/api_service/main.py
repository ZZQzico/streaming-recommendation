from fastapi import FastAPI, Response
from pydantic import BaseModel
from datetime import datetime
import sys
import os
import logging
import random
import time
import redis
import json

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # Load parent module
from kafka_stream.producer import send_interest_profile
from api_service.behavior_lookup import get_user_histories, get_recent_history, user_behavior
from metrics import get_metrics, set_kafka_sent_count, REQUEST_LATENCY, ACTIVE_USERS, get_prometheus_metrics

app = FastAPI()

# Initialize Redis connection
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# Read min and max number of other users from .env
RANDOM_USER_MIN = int(os.getenv("RANDOM_USER_MIN", 30))
RANDOM_USER_MAX = int(os.getenv("RANDOM_USER_MAX", 60))

class UserTimestampRequest(BaseModel):
    user_id: str
    timestamp: int  # Unix timestamp

class TimestampRequest(BaseModel):
    timestamp: int

class FeedbackRequest(BaseModel):
    user_id: str
    item_id: str
    is_satisfied: bool
    timestamp: int = None  # Optional, will use current time if not provided

# === Single user request endpoint ===
@app.post("/push_interest/")
def push_interest(req: UserTimestampRequest):
    # Start measuring execution time
    start_time = time.time()
    
    history_items = get_recent_history(req.user_id, req.timestamp)

    if not history_items:
        logging.warning(f"[Single] No history behavior: user={req.user_id}, ts={req.timestamp}")
        return {"status": "error", "message": "No history found"}

    send_interest_profile(req.user_id, history_items)
    set_kafka_sent_count(1)
    
    # Record active user
    ACTIVE_USERS.inc()
    
    # Record request duration
    REQUEST_LATENCY.labels(endpoint='push_interest').observe(time.time() - start_time)
    
    logging.info(f"[Single] Sent user={req.user_id}, history_len={len(history_items)}")
    return {"status": "success", "user_id": req.user_id, "history_items_count": len(history_items)}

# === Multi-user batch simulation request ===
@app.post("/send_kafka/")
def send_kafka(req: UserTimestampRequest):
    # Start measuring execution time
    start_time = time.time()
    
    sent = 0

    # 1. Main user
    history_items = get_recent_history(req.user_id, req.timestamp)
    if history_items:
        send_interest_profile(req.user_id, history_items)
        sent += 1
        
        # Record active user
        ACTIVE_USERS.inc()
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
    
    # Record request duration
    REQUEST_LATENCY.labels(endpoint='send_kafka').observe(time.time() - start_time)

    logging.info(f"[Multi] Main user + {len(random_users)} other users, sent {sent} user profiles, timestamp={req.timestamp}")
    return {"message": f"Sent {sent} user profiles to Kafka"}

# === User feedback endpoint ===
@app.post("/feedback/")
def submit_feedback(req: FeedbackRequest):
    # Start measuring execution time
    start_time = time.time()
    
    # Use current timestamp if not provided
    timestamp = req.timestamp or int(time.time())
    
    # Store feedback in Redis
    feedback_key = f"feedback:{req.user_id}:{req.item_id}:{timestamp}"
    feedback_data = {
        "user_id": req.user_id,
        "item_id": req.item_id,
        "is_satisfied": req.is_satisfied,
        "timestamp": timestamp
    }
    
    # Store as JSON
    redis_client.set(feedback_key, json.dumps(feedback_data))
    
    # Add to a list of all feedback for this user
    user_feedback_key = f"user_feedback:{req.user_id}"
    redis_client.lpush(user_feedback_key, feedback_key)
    
    # Record request duration
    REQUEST_LATENCY.labels(endpoint='feedback').observe(time.time() - start_time)
    
    logging.info(f"[Feedback] User={req.user_id}, Item={req.item_id}, Satisfied={req.is_satisfied}")
    return {"status": "success", "message": "Feedback recorded"}

# === Monitoring endpoint for Prometheus/Grafana ===
@app.get("/metrics/")
def metrics_view():
    # Legacy metrics for backward compatibility
    return get_metrics()

# === Prometheus metrics endpoint ===
@app.get("/metrics")
def prometheus_metrics():
    metrics_data, content_type = get_prometheus_metrics()
    return Response(content=metrics_data, media_type=content_type)