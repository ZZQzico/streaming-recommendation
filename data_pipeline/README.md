# Streaming Recommendation System (Local Development Guide)

This guide provides complete instructions for setting up and running the streaming recommendation system locally using Docker Compose.

## 🔧 Prerequisites

- Docker & Docker Compose installed
- Python 3.9+ (only needed for local testing outside of containers)
- Git (to clone this repository if not already)

---

## 🚀 Build & Launch the System

```bash
docker compose up --build
```

This command builds and starts the following services:

- `fastapi-server`: FastAPI-based RESTful API service
- `kafka`: Bitnami Kafka broker
- `redis`: In-memory database
- `spark-consumer`: PySpark structured streaming job
- `kafka-init`: One-shot Kafka topic initializer (creates topic `user_behavior`)

⚠️ **ZooKeeper is NOT needed.**
> This setup uses Bitnami Kafka with **KRaft (Kafka Raft mode)**. No external ZooKeeper container is used.

### ✅ Wait until you see logs like:
```
fastapi-server  | INFO:     Started server process [8]
fastapi-server  | INFO:     Waiting for application startup.
fastapi-server  | INFO:     Application startup complete.
spark-consumer  | 25/04/30 07:43:36 INFO MicroBatchExecution: Streaming query has been idle and waiting for new data more than 10000 ms.
fastapi-server  | 加载行为数据 from /mnt/project_volume/data/excluded_users_behavior.csv ...
fastapi-server  | 行为数据加载完成，共有 371269 个用户
fastapi-server  | INFO:     192.168.65.1:51835 - "GET /docs HTTP/1.1" 200 OK
fastapi-server  | INFO:     192.168.65.1:51835 - "GET /openapi.json HTTP/1.1" 200 OK
spark-consumer  | 25/04/30 07:43:46 INFO MicroBatchExecution: Streaming query has been idle and waiting for new data more than 10000 ms.
```

These logs confirm that Kafka is ready, topic is created, and Spark is consuming.

---

## 📖 API Documentation

Once `fastapi-server` is up, visit the automatic API docs:

👉 [http://localhost:8000/docs](http://localhost:8000/docs)

There are two main endpoints:

### 1. `POST /push_interest/`
Sends a single user’s interest profile to Kafka.

#### Request Body:
```json
{
  "user_id": "A34O9UGRTZNTR8",
  "timestamp": 1456923842
}
```

#### Response:
```json
{
  "status": "success",
  "user_id": "A34O9UGRTZNTR8",
  "history_items_count": 50
}
```

#### Function:
- Look up this user's recent browsing history before the timestamp.
- Send it as an interest profile message to Kafka.

---

### 2. `POST /send_kafka/`
Sends the target user’s interest profile **plus** a batch of randomly selected other users.

#### Request Body:
```json
{
  "user_id": "A34O9UGRTZNTR8",
  "timestamp": 1492384925
}
```

#### Response:
```json
{
  "message": "已发送 31 条用户画像到 Kafka"
}
```

#### Function:
- Sends the target user's history.
- Randomly samples 30–60 other users and sends their profiles to Kafka as well.

---

## 📜 Logs & Debugging

To view logs for each service, run:

```bash
docker compose logs -f fastapi-server
docker compose logs -f spark-consumer
docker compose logs -f kafka
```

Or use Docker Desktop to inspect individual container logs.

---

## 🧠 Redis Inspection

To inspect keys written by Spark:

```bash
docker exec -it redis redis-cli
```

Then inside CLI:

```redis
KEYS *
GET user:A1Q0O0GID7G2OJ
```

Redis stores user interest vectors keyed as `user:{user_id}`.

```bash
127.0.0.1:6379> keys *
 1) "user_profile:A1JK07R0LGQYPK"
 2) "user_profile:A34O9UGRTZNTR8"
 3) "user_profile:A2SVNI4VFDJ1P3"
 4) "user_profile:A2C3JD8OJ3EOE3"
 5) "user_profile:A38MGXN11M4KSB"
 6) "user_profile:A14KBIJO0P54MZ"
 7) "user_profile:AQ3LE9PHOG9BB"
 8) "user_profile:A2J05L6X34D1PQ"
 9) "user_profile:A2NZPL6HGUHNJB"
10) "user_profile:A321T1QY9YGD2H"
11) "user_profile:A7KKGIPYSHPDS"
12) "user_profile:A3PYD9W91RBOXT"
13) "user_profile:A1P6DRQ3WSZ31M"
14) "user_profile:A1T8TZ04JSUH2H"
15) "user_profile:A22OZDM3LDMX08"
16) "user_profile:AHGEPA7TKQZQ4"
17) "user_profile:A1GXF7CGG0LICF"
18) "user_profile:A2XM8N2V5PMHDZ"
19) "user_profile:A2DCUEIQSP1JVQ"
20) "user_profile:A24XI8V6PBLB8B"
21) "user_profile:A12I8A079IZGMI"
22) "user_profile:APMS0V3LUHSRP"
23) "user_profile:ARAVMLBZ1GNBZ"
24) "user_profile:A3F9FRPWVZUK6I"
25) "user_profile:A2UE1IEPS2AFYV"
26) "user_profile:A2FGI968MEBHEJ"
27) "user_profile:A283F7RF7NO4IQ"
28) "user_profile:ABCK8YW39HFTT"
29) "user_profile:A89C0PUXSYZYA"
30) "user_profile:AZ4ZGDLGJZH3"
31) "user_profile:AQ93X5F2QSV1D"
```
---

## 📈 Kafka Verification (Optional)

To check if messages are being published:

```bash
docker exec -it kafka kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic user_behavior \
  --from-beginning
```

---

## ✅ How to Confirm the Pipeline Works

The system is working if:

1. Visiting [http://localhost:8000/docs](http://localhost:8000/docs) returns the FastAPI Swagger UI.
2. Calling `/send_kafka/` logs "X 条用户画像已发送" in `fastapi-server`.
3. `spark-consumer` shows `numInputRows > 0` in structured streaming logs.
4. Redis shows new keys using `KEYS *` after sending requests.

---

## 🧼 Cleanup

To stop and remove all containers:

```bash
docker compose down
```

To also remove built images:

```bash
docker compose down --rmi all
```

---

## 🤝 Contribution

For local development and debugging, feel free to modify the `.env` file or individual Python components under:

- `data_pipeline/api_service/`
- `data_pipeline/kafka_stream/`
- `data_pipeline/config/`
- `data_pipeline/metrics/`

Then rebuild with:

```bash
docker compose up --build
```

---

Happy streaming! 🎬📡