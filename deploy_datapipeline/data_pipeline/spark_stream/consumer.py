from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, IntegerType

import os, json, sys, logging
from dotenv import load_dotenv

# Add parent directory to path to import config and metrics
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import Redis and Kafka configuration
from config.redis_config import get_redis_connection
from config.kafka_config import KAFKA_BROKER, KAFKA_TOPIC
from metrics import increment_spark_consumed, increment_redis_write

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')

# Load environment variables (e.g., checkpoint path)
load_dotenv()
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "./checkpoint")

# Define schema for incoming Kafka JSON messages
schema = StructType([
    StructField("user_id", StringType()),                     # Unique user ID
    StructField("history_items", ArrayType(StringType())),    # List of item IDs (click/view history)
    StructField("timestamp", IntegerType())                   # Timestamp used for versioning
])

# Initialize SparkSession with Kafka package
spark = SparkSession.builder \
    .appName("KafkaToRedis") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
    .getOrCreate()

# Read from Kafka topic as a streaming DataFrame
kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_BROKER) \
    .option("subscribe", KAFKA_TOPIC) \
    .option("startingOffsets", "latest") \
    .load()

# Parse Kafka value (JSON string) to structured columns
parsed_df = kafka_df.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*")

# Process each micro-batch and write data into Redis
def process_batch(df, epoch_id):
    """
    For each batch of user behavior profiles:
    - Collect rows from Spark DataFrame
    - Write each profile to Redis using key format: user_profile:{user_id}:{timestamp}
    - Update monitoring metrics
    """
    records = df.collect()
    r = get_redis_connection()

    for row in records:
        user_id = row['user_id']
        timestamp = row['timestamp']
        history_items = row['history_items']

        if user_id and timestamp and history_items:
            redis_key = f"user_profile:{user_id}:{timestamp}"
            r.set(redis_key, json.dumps(history_items))
            increment_spark_consumed()
            increment_redis_write()
            logging.info(f"[Spark] Stored {redis_key} in Redis with {len(history_items)} items")

# Start Spark structured streaming
query = parsed_df.writeStream \
    .foreachBatch(process_batch) \
    .option("checkpointLocation", CHECKPOINT_PATH) \
    .start()

query.awaitTermination()