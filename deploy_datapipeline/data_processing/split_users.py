import json
import random
import os

# Define the file path to the raw review dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, '../raw_data/Books_5.json')

# Extract all unique user IDs from the dataset (processed line-by-line)
print("Starting to extract user IDs ...")
user_set = set()
with open(file_path, 'r') as f:
    for idx, line in enumerate(f, 1):
        review = json.loads(line)
        user_set.add(review['reviewerID'])  # Add reviewer ID to set

        if idx % 1000000 == 0:
            print(f"Processed {idx} reviews, current unique user count: {len(user_set)}")

print(f"User ID extraction completed. Total unique users: {len(user_set)}")

# Shuffle the user list randomly to ensure unbiased splitting
user_list = list(user_set)
random.shuffle(user_list)

# Split users into 80% training set and 20% hold-out (excluded)
split_index = int(0.8 * len(user_list))
train_users = set(user_list[:split_index])
excluded_users = set(user_list[split_index:])

print(f"Number of training users: {len(train_users)}")
print(f"Number of excluded (hold-out) users: {len(excluded_users)}")

# Save the excluded users to excluded_users.txt for later use
excluded_users_path = os.path.join(base_dir, 'excluded_users.txt')
with open(excluded_users_path, 'w') as f:
    for user in excluded_users:
        f.write(user + '\n')

print(f"excluded_users.txt has been saved to {excluded_users_path}")