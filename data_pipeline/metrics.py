from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client.core import CollectorRegistry
import time

# Create registry
registry = CollectorRegistry()

# Counters
KAFKA_SENT_COUNT = Counter('kafka_sent_count_total', 'Number of records sent to Kafka', registry=registry)
SPARK_CONSUMED_COUNT = Counter('spark_consumed_count_total', 'Number of records consumed by Spark', registry=registry)
REDIS_WRITE_COUNT = Counter('redis_write_count_total', 'Number of records written to Redis', registry=registry)

# Histograms for latency tracking
REQUEST_LATENCY = Histogram('request_processing_seconds', 'Time spent processing request', 
                          ['endpoint'], registry=registry)

# Gauges for current values
ACTIVE_USERS = Gauge('active_users', 'Number of active users in the past minute', registry=registry)

# For backwards compatibility with existing code
kafka_sent_count = 0
spark_consumed_count = 0
redis_write_count = 0

def increment_spark_consumed():
    global spark_consumed_count
    spark_consumed_count += 1
    SPARK_CONSUMED_COUNT.inc()

def increment_redis_write():
    global redis_write_count
    redis_write_count += 1
    REDIS_WRITE_COUNT.inc()

def set_kafka_sent_count(val):
    global kafka_sent_count
    diff = val - kafka_sent_count if val > kafka_sent_count else val
    KAFKA_SENT_COUNT.inc(diff)
    kafka_sent_count = val

def get_metrics():
    return {
        "kafka_sent_count": kafka_sent_count,
        "spark_consumed_count": spark_consumed_count,
        "redis_write_count": redis_write_count
    }

def get_prometheus_metrics():
    """Return all metrics in Prometheus format."""
    return generate_latest(registry), CONTENT_TYPE_LATEST