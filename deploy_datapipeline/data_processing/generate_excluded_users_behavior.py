import json
import os

# Configure file paths relative to the current script directory
base_dir = os.path.dirname(os.path.abspath(__file__))
raw_data_path = os.path.join(base_dir, '../raw_data/Books_5.json')  # Raw Amazon review data
excluded_users_path = os.path.join(base_dir, 'excluded_users.txt')  # List of users to exclude
output_behavior_path = os.path.join(base_dir, 'excluded_users_behavior.csv')  # Output path for extracted behaviors

# Load the list of excluded users into a set for fast lookup
print("Loading excluded_users.txt ...")
excluded_users = set()
with open(excluded_users_path, 'r') as f:
    for line in f:
        excluded_users.add(line.strip())  # Remove newline and add user_id to the set

print(f"Number of excluded users: {len(excluded_users)}")

# Parse Books_5.json line-by-line and extract interactions of excluded users
print("Extracting behaviors of excluded users ...")
behaviors = []

with open(raw_data_path, 'r') as f:
    for idx, line in enumerate(f, 1):
        review = json.loads(line)  # Parse the JSON line
        user_id = review['reviewerID']  # User identifier
        item_id = review['asin']       # Item identifier (ASIN)
        timestamp = review['unixReviewTime']  # Timestamp in Unix format (seconds)

        # If the user is in the excluded list, collect their behavior
        if user_id in excluded_users:
            behaviors.append((user_id, item_id, timestamp))

        if idx % 1000000 == 0:
            print(f"Processed {idx} reviews")

# Sort the extracted behaviors by user_id and timestamp
print(f"Total extracted behaviors from excluded users: {len(behaviors)}")
behaviors.sort(key=lambda x: (x[0], x[2]))  # Sort by (user_id, timestamp)

# Write the behavior data to a CSV file with header
with open(output_behavior_path, 'w') as f:
    f.write('user_id,item_id,timestamp\n')
    for user_id, item_id, timestamp in behaviors:
        f.write(f"{user_id},{item_id},{timestamp}\n")

print(f"excluded_users_behavior.csv has been saved to {output_behavior_path}")