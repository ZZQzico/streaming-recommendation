# 🎯 Streaming Recommendation System — Full Guide

This README provides everything you need to:

- Run the system locally with Docker Compose
- Interact with RESTful APIs
- Inspect Redis and Kafka
- Deploy and test the system on Chameleon Cloud (FastAPI hosted at `129.114.25.200`)

---

## 🛠️ Local Development

### Prerequisites

- Docker & Docker Compose
- Python 3.9+ (optional for local-only testing)
- Git

### Build & Run Locally

```bash
docker compose up --build
```

Services launched:

- `fastapi-server`: RESTful API with Kafka producer and Redis integration
- `kafka`: Bitnami Kafka in KRaft (no Zookeeper needed)
- `redis`: Caching user interest profiles and recommendation results
- `spark-consumer`: PySpark structured streaming to consume from Kafka and write to Redis
- `kafka-init`: Creates topic `user_behavior`

### View Logs

```bash
docker compose logs -f fastapi-server
docker compose logs -f spark-consumer
```

### Local Swagger Docs

Access locally at:

```
http://localhost:8000/docs
```

---

## ☁️ Cloud Deployment (Chameleon)

### FastAPI Server

- Public URL: `http://129.114.25.200:8000`
- Swagger Docs: [http://129.114.25.200:8000/docs](http://129.114.25.200:8000/docs)

### Test via cURL

```bash
curl -X POST http://129.114.25.200:8000/send_kafka/ \
  -H "Content-Type: application/json" \
  -d '{"user_id": "A035230154WEA8JCP8HS", "timestamp": 1609459200}'
```

Expected response:

```json
{"message":"Sent 31 user profiles to Kafka"}
```

---

## 🧠 Inspect Redis

### Local Access

```bash
docker exec -it redis redis-cli
KEYS *
GET user_profile:A123456789:1609459200
```

### Access Redis from VM (e.g. Redis running on `192.168.1.11`)

```bash
redis-cli -h 192.168.1.11
GET recommendation_result:A123456789:1609459200
```

### Manually Insert Data into Redis

```python
import redis, json
r = redis.Redis(host='192.168.1.11', port=6379)
key = "recommendation_result:A123456789:1609459200"
value = json.dumps(["B001", "B002", "B003"])
r.set(key, value)
```

---

## 📖 Full API Reference

All APIs are documented and testable at:

```
http://localhost:8000/docs
```

or (cloud):

```
http://129.114.25.200:8000/docs
```

---

### 1. `POST /push_interest/`

Send a single user's interest profile to Kafka.

#### Request

```json
{
  "user_id": "A123456789",
  "timestamp": 1609459200
}
```

#### cURL

```bash
curl -X POST http://localhost:8000/push_interest/ \
  -H "Content-Type: application/json" \
  -d '{"user_id": "A123456789", "timestamp": 1609459200}'
```

---

### 2. `POST /send_kafka/`

Send the target user + 30–60 randomly selected users to Kafka.

#### Request

```json
{
  "user_id": "A123456789",
  "timestamp": 1609459200
}
```

#### cURL

```bash
curl -X POST http://localhost:8000/send_kafka/ \
  -H "Content-Type: application/json" \
  -d '{"user_id": "A123456789", "timestamp": 1609459200}'
```

---

### 3. `POST /recommendation/`

Manually store a recommendation result into Redis.

#### Request

```json
{
  "user_id": "A123456789",
  "timestamp": 1609459200,
  "recommendation": ["B001", "B002", "B003"]
}
```

#### cURL

```bash
curl -X POST http://localhost:8000/recommendation/ \
  -H "Content-Type: application/json" \
  -d '{"user_id": "A123456789", "timestamp": 1609459200, "recommendation": ["B001", "B002"]}'
```

---

### 4. `GET /recommendation/{user_id}/{timestamp}`

Query a recommendation result from Redis.

#### cURL

```bash
curl http://localhost:8000/recommendation/A123456789/1609459200
```

---

### 5. `GET /user_profile/{user_id}/{timestamp}`

Retrieve a user’s interest profile.

#### cURL

```bash
curl http://localhost:8000/user_profile/A123456789/1609459200
```

---

### 6. `GET /metrics/`

Returns internal counters (e.g. Kafka sent count).

#### cURL

```bash
curl http://localhost:8000/metrics/
```

---

## ✅ End-to-End Verification Checklist

| Step | Expected Result |
|------|------------------|
| Swagger UI loads | `http://localhost:8000/docs` |
| `/send_kafka/` returns success | Kafka log shows sent messages |
| `spark-consumer` logs | Show `numInputRows > 0` |
| Redis | Keys appear under `user_profile:*:*` |
| Recommendation endpoints | Return stored data or correct error |

---

Happy building and deploying! 🎬📡