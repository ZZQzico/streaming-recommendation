import os
import json
import logging
import asyncio
from fastapi import FastAPI, BackgroundTasks
from dotenv import load_dotenv
import redis
from models import RecommendationModel
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv()

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

# Create FastAPI application
app = FastAPI(title="Recommendation System Model Service")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Create Redis connection
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    decode_responses=True
)

# Load recommendation model
recommendation_model = RecommendationModel()

# Background task to process a single user
async def process_user(user_id):
    # Get user history from Redis
    user_key = f"user_profile:{user_id}"
    user_history = redis_client.get(user_key)
    
    if user_history:
        history_items = json.loads(user_history)
        # Generate recommendations using the model
        recommendations = recommendation_model.predict(user_id, history_items)
        
        # Store recommendations in Redis
        recommendations_key = f"recommendations:{user_id}"
        redis_client.set(recommendations_key, json.dumps(recommendations))
        logging.info(f"Recommendations for user {user_id} saved to Redis, total {len(recommendations)} items")
    else:
        logging.warning(f"No history data found for user {user_id}")

# Background task: continuously monitor Redis for user data and generate recommendations
async def continuous_monitoring():
    logging.info("Starting continuous monitoring task...")
    while True:
        try:
            # Scan all user profiles
            user_keys = redis_client.keys("user_profile:*")
            for user_key in user_keys:
                user_id = user_key.split(":")[-1]
                await process_user(user_id)
                
            # Check every second
            await asyncio.sleep(1)
        except Exception as e:
            logging.error(f"Monitoring task error: {str(e)}")
            await asyncio.sleep(5)  # Wait longer after an error

@app.on_event("startup")
async def startup_event():
    # Ensure models are loaded
    if not recommendation_model.is_ready():
        recommendation_model.load_models()
    
    # Start continuous monitoring of Redis
    logging.info("Model service started, monitoring Redis for user data...")
    asyncio.create_task(continuous_monitoring())

@app.on_event("shutdown")
async def shutdown_event():
    logging.info("Model service shutting down...")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": recommendation_model.is_ready()
    }

@app.get("/metrics")
async def get_metrics():
    metrics = {
        "redis_keys": len(redis_client.keys("*")),
        "user_profiles": len(redis_client.keys("user_profile:*")),
        "recommendations": len(redis_client.keys("recommendations:*"))
    }
    return metrics

@app.post("/recommend/{user_id}")
async def recommend(user_id: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_user, user_id)
    return {"status": "success", "message": f"Started generating recommendations for user {user_id}"}

@app.get("/recommendations/{user_id}")
async def get_recommendations(user_id: str):
    # Get recommendations from Redis
    recommendations_key = f"recommendations:{user_id}"
    recommendations_json = redis_client.get(recommendations_key)
    
    if recommendations_json:
        recommendations = json.loads(recommendations_json)
        return {"status": "success", "recommendations": recommendations}
    else:
        return {"status": "error", "message": "No recommendations found"} 