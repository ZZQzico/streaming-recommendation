import json
import csv
import os

# Define file paths
base_dir = os.path.dirname(os.path.abspath(__file__))
books_path = os.path.join(base_dir, '../raw_data/Books_5.json')  # Path to raw review data
excluded_users_path = os.path.join(base_dir, 'excluded_users.txt')  # Users to exclude from training
lightgcn_data_path = os.path.join(base_dir, 'lightgcn_data.csv')  # Output path for LightGCN training data

# Load excluded users from file into a set
print("Loading excluded_users.txt ...")
excluded_users = set()
with open(excluded_users_path, 'r') as f:
    for line in f:
        excluded_users.add(line.strip())

print(f"Loaded {len(excluded_users)} excluded users")

# Process Books_5.json line-by-line to extract training samples for LightGCN
print("Processing Books_5.json to generate LightGCN training samples ...")
lightgcn_rows = set()
with open(books_path, 'r') as f:
    for idx, line in enumerate(f, 1):
        review = json.loads(line)
        user_id = review['reviewerID']
        item_id = review['asin']
        rating = review['overall']

        # Include only reviews with rating >= 4 from non-excluded users
        if rating >= 4 and user_id not in excluded_users:
            lightgcn_rows.add((user_id, item_id))

        if idx % 1000000 == 0:
            print(f"Processed {idx} reviews, current LightGCN sample count: {len(lightgcn_rows)}")

print(f"Completed processing Books_5.json, total LightGCN samples generated: {len(lightgcn_rows)}")

# Write the LightGCN training samples to CSV
print(f"Saving to {lightgcn_data_path} ...")
with open(lightgcn_data_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['user_id', 'item_id'])  # Write CSV header
    for idx, (user_id, item_id) in enumerate(lightgcn_rows, 1):
        writer.writerow([user_id, item_id])
        if idx % 1000000 == 0:
            print(f"Wrote {idx} LightGCN samples ...")

print("LightGCN training data generation completed!")