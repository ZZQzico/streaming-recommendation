import json
import os
import time
from kafka import KafkaProducer, KafkaAdminClient
from kafka.admin import NewTopic
from kafka.errors import TopicAlreadyExistsError, NoBrokersAvailable
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv()

# Kafka configuration - Using Docker service name
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")  # Modified to use kafka container name
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "user_behavior")


def wait_for_kafka(timeout=60):
    """Wait for Kafka broker to be available"""
    start = time.time()
    while True:
        try:
            KafkaAdminClient(bootstrap_servers=KAFKA_BROKER).list_topics()
            logging.info("[Kafka] Kafka broker is ready")
            return
        except NoBrokersAvailable:
            if time.time() - start > timeout:
                logging.error("[Kafka] Waiting for Kafka broker timed out")
                raise
            logging.info("[Kafka] Waiting for Kafka broker to start...")
            time.sleep(2)


def create_topic_if_not_exists(topic_name):
    """Ensure Topic exists, create if not"""
    admin = KafkaAdminClient(bootstrap_servers=KAFKA_BROKER)
    try:
        admin.create_topics([NewTopic(name=topic_name, num_partitions=1, replication_factor=1)])
        logging.info(f"[Kafka] Topic '{topic_name}' created")
    except TopicAlreadyExistsError:
        logging.info(f"[Kafka] Topic '{topic_name}' already exists")


# Wait for Kafka to start
try:
    wait_for_kafka()

    # Ensure topic exists
    create_topic_if_not_exists(KAFKA_TOPIC)

    # Kafka Producer instance
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )
    
    logging.info(f"[Kafka] Successfully connected to Kafka: {KAFKA_BROKER}")
except Exception as e:
    logging.error(f"[Kafka] Initialization failed: {e}")
    # Create an empty producer to avoid startup errors
    producer = None


def send_interest_profile(user_id, history_items):
    if producer is None:
        logging.error("[Kafka] Cannot send message: Kafka connection not initialized")
        return {"status": "error", "message": "Kafka connection not initialized"}
    
    message = {
        "user_id": user_id,
        "history_items": history_items
    }
    try:
        producer.send(KAFKA_TOPIC, message)
        producer.flush()
        logging.info(f"[Kafka] Sent user_id={user_id}, items={history_items}")
        return {"status": "success", "message": "Message sent to Kafka"}
    except Exception as e:
        logging.error(f"[Kafka] Send failed: {e}")
        return {"status": "error", "message": f"Send failed: {str(e)}"}


# Support CLI test call
if __name__ == "__main__":
    test_user = "test_user_001"
    test_history = ["itemA", "itemB", "itemC"]
    send_interest_profile(test_user, test_history)