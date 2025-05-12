import os
import csv
import logging
from bisect import bisect_right
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)

WINDOW_SIZE = int(os.getenv('WINDOW_SIZE', 1))  # Default window=1

# Preload all excluded_users_behavior
user_behavior = defaultdict(list)

# Define possible file paths (Docker environment and local environment)
docker_path = '/mnt/project_volume/data/excluded_users_behavior.csv'
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_dir, "../.."))
local_path = os.path.join(project_root, 'data_processing/excluded_users_behavior.csv')
alternate_path = os.path.join(project_root, 'data_processing', 'excluded_users_behavior.csv')

# Try to load behavior data
behavior_csv_path = os.getenv('BEHAVIOR_CSV_PATH', '')

# If environment variable is not set, try all possible paths
if not behavior_csv_path:
    possible_paths = [docker_path, local_path, alternate_path]
    for path in possible_paths:
        if os.path.exists(path):
            behavior_csv_path = path
            break

logging.info(f"Trying to load behavior data from {behavior_csv_path}...")

# Provide sample data (to prevent service failure if file doesn't exist)
sample_data = [
    ("A2SRVEKZS33OGN", [("0764227971", 1492384925), ("0990351408", 1492384920), 
                         ("1492635227", 1492384915), ("0099740915", 1492384910), ("0764213466", 1492384905)]),
    ("A15AFEWW8CUDBT", [("B000FA64PQ", 1492384925), ("B0006HBAJO", 1492384920), 
                         ("B00004U9V2", 1492384915), ("B00008RW9U", 1492384910), ("B0000AC9ML", 1492384905)]),
    ("A18CL73MAOOP78", [("B000SE5SY6", 1492384925), ("B0000TWUT4", 1492384920), 
                         ("B0006GUZ7Q", 1492384915), ("B00004UE6G", 1492384910), ("B00006HBB6", 1492384905)])
]

try:
    # Try to load data from file
    if behavior_csv_path and os.path.exists(behavior_csv_path):
        with open(behavior_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                user_id = row['user_id']
                item_id = row['item_id']
                timestamp = int(row['timestamp'])
                user_behavior[user_id].append((timestamp, item_id))
        logging.info(f"Behavior data loading completed, total {len(user_behavior)} users")
    else:
        # Use sample data
        logging.warning(f"Could not find behavior data file, using sample data instead")
        for user_id, items in sample_data:
            for item_id, timestamp in items:
                user_behavior[user_id].append((timestamp, item_id))
        logging.info(f"Sample data loading completed, total {len(user_behavior)} users")
        
    # Sort each user's data by time
    for user_id in user_behavior:
        user_behavior[user_id].sort()
        
except Exception as e:
    logging.error(f"Error loading behavior data: {str(e)}")
    # Use sample data
    logging.warning(f"Due to error, using sample data instead")
    for user_id, items in sample_data:
        for item_id, timestamp in items:
            user_behavior[user_id].append((timestamp, item_id))
    
    # Sort each user's data by time
    for user_id in user_behavior:
        user_behavior[user_id].sort()

# single user
def get_recent_history(user_id, timestamp, max_history=50):
    if user_id not in user_behavior:
        logging.warning(f"No history behavior found for user {user_id}")
        return []

    behaviors = user_behavior[user_id]
    timestamps = [t for t, _ in behaviors]
    idx = bisect_right(timestamps, timestamp)

    selected = behaviors[:idx][-max_history:]
    history_items = [item_id for _, item_id in selected]
    return history_items

#multi users
def get_user_histories(target_ts, max_history=50, window_size=WINDOW_SIZE):
    user_histories = defaultdict(list)

    for user_id, behaviors in user_behavior.items():
        for ts, item_id in behaviors:
            if abs(ts - target_ts) <= window_size:
                user_histories[user_id].append((ts, item_id))

    result = {}
    for user_id, records in user_histories.items():
        sorted_items = sorted(records, key=lambda x: x[0])
        result[user_id] = [item_id for _, item_id in sorted_items[-max_history:]]

    return result  # Returns dict[user_id] = [item_id1, item_id2, ...]