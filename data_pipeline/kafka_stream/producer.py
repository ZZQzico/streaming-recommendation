import json
import os
import time
from kafka import KafkaProducer, KafkaAdminClient
from kafka.admin import NewTopic
from kafka.errors import TopicAlreadyExistsError, NoBrokersAvailable
from dotenv import load_dotenv
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)

# 加载环境变量
load_dotenv()

# Kafka 配置
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "user_behavior")


def wait_for_kafka(timeout=60):
    """等待 Kafka broker 启动可连接"""
    start = time.time()
    while True:
        try:
            KafkaAdminClient(bootstrap_servers=KAFKA_BROKER).list_topics()
            logging.info("[Kafka] Kafka broker 已就绪")
            return
        except NoBrokersAvailable:
            if time.time() - start > timeout:
                logging.error("[Kafka] 等待 Kafka broker 超时")
                raise
            logging.info("[Kafka] 等待 Kafka broker 启动中...")
            time.sleep(2)


def create_topic_if_not_exists(topic_name):
    """确保 Topic 存在，不存在就创建"""
    admin = KafkaAdminClient(bootstrap_servers=KAFKA_BROKER)
    try:
        admin.create_topics([NewTopic(name=topic_name, num_partitions=1, replication_factor=1)])
        logging.info(f"[Kafka] Topic '{topic_name}' 已创建")
    except TopicAlreadyExistsError:
        logging.info(f"[Kafka] Topic '{topic_name}' 已存在")


# 等待 Kafka 启动
wait_for_kafka()

# 确保 topic 存在
create_topic_if_not_exists(KAFKA_TOPIC)

# Kafka Producer 实例
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)


def send_interest_profile(user_id, history_items):
    message = {
        "user_id": user_id,
        "history_items": history_items
    }
    try:
        producer.send(KAFKA_TOPIC, message)
        producer.flush()
        logging.info(f"[Kafka] 已发送 user_id={user_id}, items={history_items}")
    except Exception as e:
        logging.error(f"[Kafka] 发送失败: {e}")


# 支持 CLI 测试调用
if __name__ == "__main__":
    test_user = "test_user_001"
    test_history = ["itemA", "itemB", "itemC"]
    send_interest_profile(test_user, test_history)