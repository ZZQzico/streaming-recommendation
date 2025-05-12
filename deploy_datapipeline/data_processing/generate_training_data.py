import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random
from collections import defaultdict
import csv
import os
import time

# Define paths to raw data and output files
base_dir = os.path.dirname(os.path.abspath(__file__))
books_path = os.path.join(base_dir, '../raw_data/Books_5.json')          # Review data
meta_path = os.path.join(base_dir, '../raw_data/meta_Books.json')        # Metadata for items
excluded_users_path = os.path.join(base_dir, 'excluded_users.txt')       # Test users to exclude

# === Step 1: Load excluded users ===
print("Loading excluded_users.txt ...")
excluded_users = set()
with open(excluded_users_path, 'r') as f:
    for line in f:
        excluded_users.add(line.strip())
print(f"Loaded {len(excluded_users)} excluded users (20% hold-out)")

# === Step 2: Parse meta_Books.json to generate item embeddings ===
print("Processing meta_Books.json to generate item embeddings ...")
item_embeddings = {}
scaler = MinMaxScaler()

# First pass: collect all prices for normalization
prices = []
with open(meta_path, 'r') as f:
    for line in f:
        item = json.loads(line)
        price = item.get('price', 0.0)
        if isinstance(price, str):
            price = price.replace('$', '').replace(',', '').strip()
        try:
            price = float(price)
            prices.append(price)
        except ValueError:
            continue

if prices:
    prices = np.array(prices).reshape(-1, 1)
    scaler.fit(prices)

# Second pass: compute embeddings per item
with open(meta_path, 'r') as f:
    for idx, line in enumerate(f, 1):
        item = json.loads(line)
        item_id = item['asin']
        
        # Process category
        categories_raw = item.get('categories', [['Unknown']])
        categories_last = [path[-1] for path in categories_raw if path]
        category_combined = '_'.join(categories_last)
        category_hash = hash(category_combined) % 100 / 100

        # Process brand
        brand = item.get('brand', 'Unknown')
        brand_hash = hash(brand) % 100 / 100

        # Process price
        price = item.get('price', 0.0)
        if isinstance(price, str):
            price = price.replace('$', '').replace(',', '').strip()
        try:
            price = float(price)
            price_scaled = scaler.transform([[price]])[0][0]
        except ValueError:
            price_scaled = 0.0

        # Combine into embedding
        embedding = [category_hash, brand_hash, price_scaled]
        item_embeddings[item_id] = embedding

        if idx % 100000 == 0:
            print(f"Processed {idx} items, current embedding count: {len(item_embeddings)}")

print(f"Generated {len(item_embeddings)} item embeddings")

# === Step 3: Build user interaction histories ===
print("Building user interaction histories ...")
user_histories = defaultdict(list)
all_items = set()

with open(books_path, 'r') as f:
    for idx, line in enumerate(f, 1):
        review = json.loads(line)
        user_id = review['reviewerID']
        if user_id in excluded_users:
            continue

        item_id = review['asin']
        rating = review['overall']
        timestamp = review['unixReviewTime']

        user_histories[user_id].append((timestamp, item_id, rating))
        all_items.add(item_id)

        if idx % 1000000 == 0:
            print(f"Processed {idx} reviews, current user count: {len(user_histories)}")

# Sort each user's history chronologically
for user_id in user_histories:
    user_histories[user_id].sort()

print(f"Completed user history collection. Total users: {len(user_histories)}, total items: {len(all_items)}")

# === NOTE: Sample tuning parameters for balancing training data ===
# You can adjust:
# - `user_sample_rate_high` and `user_sample_rate_low` to control sampling for high/low activity users
# - `neg_sample_num` to change the ratio of positive/negative samples

# === Step 4: Generate training data and write to CSV ===
# Configuration
neg_sample_num = 2                # Number of negative samples per positive
flush_interval = 1000            # Batch write threshold
user_sample_rate_high = 0.5      # Sampling rate for users with many interactions
user_sample_rate_low = 0.1       # Sampling rate for users with few interactions
min_history_len = 10             # Minimum interactions for high-quality users
max_history_len = 50             # Max history length to retain
history_step = 2                 # Stride for sliding window

print("Generating training data and writing to train_data.csv ...")
train_data_path = os.path.join(base_dir, 'train_data.csv')

with open(train_data_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['user_id', 'history_items', 'candidate_item', 'label'])

    buffer_rows = []
    processed_users = 0

    for u_idx, (user_id, interactions) in enumerate(user_histories.items(), 1):
        interactions_sorted = sorted(interactions)
        interaction_count = len(interactions_sorted)

        # Decide whether to include this user based on sampling strategy
        if interaction_count >= min_history_len:
            if random.random() > user_sample_rate_high:
                continue
            start_idx = min_history_len
        else:
            if random.random() > user_sample_rate_low:
                continue
            start_idx = 1

        start_time = time.time()

        # Get negative sampling pool
        user_items = set([item for (_, item, _) in interactions_sorted])
        neg_items_pool = list(all_items - user_items)

        if len(neg_items_pool) > 5000:
            neg_items_pool = random.sample(neg_items_pool, 5000)

        # Truncate history if too long
        if interaction_count > max_history_len:
            interactions_sorted = interactions_sorted[-max_history_len:]

        # Generate samples with sliding window
        for idx in range(start_idx, len(interactions_sorted), history_step):
            history = [item for (_, item, _) in interactions_sorted[:idx]]
            if len(history) == 0:
                continue

            pos_item = interactions_sorted[idx][1]
            rating = interactions_sorted[idx][2]

            if rating >= 4:
                buffer_rows.append([user_id, '|'.join(history), pos_item, 1])  # Positive sample

                # Add negative samples
                if neg_items_pool:
                    neg_samples = random.sample(neg_items_pool, min(neg_sample_num, len(neg_items_pool)))
                    for neg_item in neg_samples:
                        buffer_rows.append([user_id, '|'.join(history), neg_item, 0])  # Negative sample

            # Write in batches to save memory
            if len(buffer_rows) >= flush_interval:
                writer.writerows(buffer_rows)
                buffer_rows.clear()

        processed_users += 1

        if processed_users % 1000 == 0:
            print(f"Processed {processed_users} users, elapsed: {time.time() - start_time:.2f}s")

    if buffer_rows:
        writer.writerows(buffer_rows)

print(f"Finished writing training data. Total users processed: {processed_users}")

# === Step 5: Save item embeddings to CSV ===
print("Saving item_embeddings.csv ...")
item_embedding_path = os.path.join(base_dir, 'item_embeddings.csv')
with open(item_embedding_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['item_id', 'category_hash', 'brand_hash', 'price_scaled'])  # Header row
    for item_id, embedding in item_embeddings.items():
        writer.writerow([item_id] + embedding)

print("Item embeddings saved successfully.")