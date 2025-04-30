import os
from dotenv import load_dotenv

load_dotenv()

# Kafka配置
KAFKA_BROKER = os.getenv('KAFKA_BROKER', 'localhost:9092')
KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'user_behavior')