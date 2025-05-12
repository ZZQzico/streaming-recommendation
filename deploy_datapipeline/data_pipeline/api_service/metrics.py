import time
from fastapi import Response
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

# -------- Prometheus Metrics --------
# Kafka 发送计数（Gauge，因为设置为指定值）
kafka_sent_count = Gauge(
    "kafka_sent_count", "Number of Kafka messages sent"
)
# Spark 消费计数
spark_consumed_count = Counter(
    "spark_consumed_count", "Number of records consumed by Spark"
)
# Redis 写入计数
redis_write_count = Counter(
    "redis_write_count", "Number of write operations to Redis"
)

# HTTP 请求延迟（按 endpoint 分标签）
req_latency = Histogram(
    "req_latency_seconds", "Latency of HTTP requests",
    ["endpoint"]
)
# HTTP 错误计数
req_errors = Counter(
    "req_errors_total", "Count of HTTP request errors",
    ["endpoint"]
)

# -------- Business Metrics Functions --------
def increment_spark_consumed():
    """Increment Spark consumed counter"""
    spark_consumed_count.inc()

def increment_redis_write():
    """Increment Redis write counter"""
    redis_write_count.inc()

def set_kafka_sent_count(val: int):
    """Set Kafka sent count gauge to specified value"""
    kafka_sent_count.set(val)

# -------- Expose Metrics Endpoint --------
def get_metrics():
    """Return Prometheus metrics in HTTP response"""
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
