kafka_sent_count = 0
spark_consumed_count = 0
redis_write_count = 0

def increment_spark_consumed():
    global spark_consumed_count
    spark_consumed_count += 1

def increment_redis_write():
    global redis_write_count
    redis_write_count += 1

def set_kafka_sent_count(val):
    global kafka_sent_count
    kafka_sent_count = val

def get_metrics():
    return {
        "kafka_sent_count": kafka_sent_count,
        "spark_consumed_count": spark_consumed_count,
        "redis_write_count": redis_write_count
    }