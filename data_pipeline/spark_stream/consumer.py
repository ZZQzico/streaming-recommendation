from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, ArrayType

import redis
import json
import os
import sys
from dotenv import load_dotenv
import logging

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)

# 加载环境变量
load_dotenv()

# 添加模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 导入自定义模块
from config.redis_config import get_redis_connection
from config.kafka_config import KAFKA_BROKER, KAFKA_TOPIC
from metrics import increment_spark_consumed, increment_redis_write

# Kafka Schema
schema = StructType([
    StructField("user_id", StringType()),
    StructField("history_items", ArrayType(StringType()))
])

# 检查点路径
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "./checkpoint")

# SparkSession
spark = SparkSession.builder \
    .appName("KafkaToRedis") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
    .getOrCreate()

# Kafka Stream
kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_BROKER) \
    .option("subscribe", KAFKA_TOPIC) \
    .option("startingOffsets", "latest") \
    .load()

# 解析 value 字段
value_df = kafka_df.selectExpr("CAST(value AS STRING)")
parsed_df = value_df.select(from_json(col("value"), schema).alias("data")).select("data.*")

# 批处理写入 Redis，并更新 metrics
def process_batch(df, epoch_id):
    records = df.collect()
    r = get_redis_connection()
    for row in records:
        user_id = row['user_id']
        history_items = row['history_items']
        if user_id and history_items:
            r.set(f"user_profile:{user_id}", json.dumps(history_items))
            increment_spark_consumed()
            increment_redis_write()
            logging.info(f"[Spark] 写入 Redis: user={user_id}, {len(history_items)} items")

# 启动流处理
query = parsed_df.writeStream \
    .foreachBatch(process_batch) \
    .start()

query.awaitTermination()