# Recommendation System Model Service

This is a FastAPI-based recommendation system model service that integrates three models (LightGCN, DIN, and RankNet) and provides recommendation functionality as a REST API.

## Features

- Retrieves user historical behavior data from Redis
- Generates personalized recommendations using multiple recommendation models
- Writes recommendation results back to Redis for frontend use
- Provides REST API endpoints for direct calling
- Supports continuous monitoring of user data in Redis
- Provides health check and performance metrics APIs

## System Requirements

- Python 3.9+
- PyTorch
- FastAPI 
- Redis
- Docker (optional, recommended)

## Quick Start

### Using Docker (Recommended)

1. Ensure Docker and docker-compose are installed
2. Build and start the model service:

```bash
docker-compose build model-service
docker-compose up -d model-service
```

### Manual Run

1. Install dependencies:

```bash
pip install torch pandas numpy redis fastapi uvicorn python-dotenv
```

2. Ensure the `.env` file contains the necessary configuration items:

```
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0
```

3. Start the service:

```bash
cd model_service
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

## Testing the Service

### Health Check

```bash
curl http://localhost:8001/health
```

Normal response:
```json
{"status":"healthy","model_loaded":true}
```

### Get Metrics

```bash
curl http://localhost:8001/metrics
```

### Manually Trigger Recommendation Generation

Trigger recommendations for a specific user:

```bash
curl -X 'POST' 'http://localhost:8001/recommend/{user_id}' -H 'accept: application/json'
```

For example:

```bash
curl -X 'POST' 'http://localhost:8001/recommend/A2SRVEKZS33OGN' -H 'accept: application/json'
```

### Test Complete Process

1. Write user history data to Redis:

```bash
docker exec -it redis redis-cli SET "user_profile:A2SRVEKZS33OGN" '["0764227971", "0990351408", "1492635227", "0099740915", "0764213466"]'
```

2. Trigger recommendation generation:

```bash
curl -X 'POST' 'http://localhost:8001/recommend/A2SRVEKZS33OGN' -H 'accept: application/json'
```

3. Get recommendation results from Redis:

```bash
docker exec -it redis redis-cli GET "recommendations:A2SRVEKZS33OGN"
```

## API Endpoints

- `GET /health`: Health check
- `GET /metrics`: Service metrics
- `POST /recommend/{user_id}`: Generate recommendations for the specified user

## Troubleshooting

1. If the service cannot connect to Redis:
   - Check if the Redis service is running
   - Verify that the Redis configuration in the environment variables is correct

2. If model loading fails:
   - Ensure the model files are in the correct path (`/app/model_output/[model_name]/`)
   - Check if the model file format is correct

3. If recommendation results are empty:
   - Check if the user history data exists in Redis
   - Verify the user ID format is correct

## Log Monitoring

View model service logs:

```bash
docker logs model-service
```

## Architecture Description

This service adopts a multi-stage recommendation architecture:
1. Use the LightGCN model for collaborative filtering to generate a broad candidate set
2. Use the DIN model for rough ranking of the candidate set
3. Use the RankNet model for fine-grained ranking of the sorted results to generate the final recommendation list 