import os
import csv
from bisect import bisect_right
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

WINDOW_SIZE = int(os.getenv('WINDOW_SIZE', 1))

# Load user behavior from CSV
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_dir, "../.."))
behavior_csv_path = os.getenv('BEHAVIOR_CSV_PATH', os.path.join(project_root, 'data_processing/excluded_users_behavior.csv'))

print(f"Loading behavior data from {behavior_csv_path} ...")
user_behavior = defaultdict(list)

with open(behavior_csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        user_id = row['user_id']
        item_id = row['item_id']
        timestamp = int(row['timestamp'])
        user_behavior[user_id].append((timestamp, item_id))

# Sort behaviors by timestamp per user
for user_id in user_behavior:
    user_behavior[user_id].sort()

print(f"Loaded {len(user_behavior)} users")

#single user behavior history
def get_recent_history(user_id, timestamp, max_history=50):
    if user_id not in user_behavior:
        return []
    behaviors = user_behavior[user_id]
    timestamps = [t for t, _ in behaviors]
    idx = bisect_right(timestamps, timestamp)
    selected = behaviors[:idx][-max_history:]
    return [item_id for _, item_id in selected]

#get user behaviors near one timestamp to simulate high qps(for test) 
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
    return result