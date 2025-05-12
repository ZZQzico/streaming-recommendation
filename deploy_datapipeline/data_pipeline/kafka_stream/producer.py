import json
import os
import time
from kafka import KafkaProducer, KafkaAdminClient
from kafka.admin import NewTopic
from kafka.errors import TopicAlreadyExistsError, NoBrokersAvailable
from dotenv import load_dotenv
import logging

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)

# Load environment variables from .env
load_dotenv()

# Kafka configuration from .env or default
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "user_behavior")


def wait_for_kafka(timeout=60):
    """Wait until Kafka broker is available or timeout."""
    start = time.time()
    while True:
        try:
            KafkaAdminClient(bootstrap_servers=KAFKA_BROKER).list_topics()
            logging.info("[Kafka] Kafka broker is available")
            return
        except NoBrokersAvailable:
            if time.time() - start > timeout:
                logging.error("[Kafka] Timed out waiting for Kafka broker")
                raise
            logging.info("[Kafka] Waiting for Kafka broker to start...")
            time.sleep(2)


def create_topic_if_not_exists(topic_name):
    """Create Kafka topic if it does not already exist."""
    admin = KafkaAdminClient(bootstrap_servers=KAFKA_BROKER)
    try:
        admin.create_topics([NewTopic(name=topic_name, num_partitions=1, replication_factor=1)])
        logging.info(f"[Kafka] Topic '{topic_name}' created")
    except TopicAlreadyExistsError:
        logging.info(f"[Kafka] Topic '{topic_name}' already exists")


# Wait for Kafka to be ready
wait_for_kafka()

# Ensure the topic is created
create_topic_if_not_exists(KAFKA_TOPIC)

# Kafka producer instance
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)


def send_interest_profile(user_id, history_items, timestamp=None):
    """
    Send a user interest profile to Kafka with optional timestamp included in key.
    If timestamp is given, it is appended to the key so that downstream consumers (e.g., Redis) can
    distinguish entries by both user and time.
    """
    message = {
        "user_id": user_id,
        "history_items": history_items,
        "timestamp": timestamp
    }
    try:
        producer.send(KAFKA_TOPIC, message)
        producer.flush()
        logging.info(f"[Kafka] Sent user_id={user_id}, items={history_items}")
    except Exception as e:
        logging.error(f"[Kafka] Failed to send message: {e}")


# Optional CLI usage for testing
if __name__ == "__main__":
    test_user = "test_user_001"
    test_history = ["itemA", "itemB", "itemC"]
    test_timestamp = 1609459200
    send_interest_profile(test_user, test_history, test_timestamp)