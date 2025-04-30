import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random
from collections import defaultdict
import csv
import os
import time

base_dir = os.path.dirname(os.path.abspath(__file__))
books_path = os.path.join(base_dir, '../raw_data/Books_5.json')
meta_path = os.path.join(base_dir, '../raw_data/meta_Books.json')
excluded_users_path = os.path.join(base_dir, 'excluded_users.txt')

# === 第一步：加载 excluded_users.txt ===
print("加载 excluded_users.txt ...")
excluded_users = set()
with open(excluded_users_path, 'r') as f:
    for line in f:
        excluded_users.add(line.strip())
print(f"排除 {len(excluded_users)} 个用户，仅保留 80% 训练用户")

# === 第二步：处理 meta_Books.json，生成 item embedding ===
print("处理 meta_Books.json，生成 item embedding ...")
item_embeddings = {}
scaler = MinMaxScaler()

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

with open(meta_path, 'r') as f:
    for idx, line in enumerate(f, 1):
        item = json.loads(line)
        item_id = item['asin']
        categories_raw = item.get('categories', [['Unknown']])
        categories_last = [path[-1] for path in categories_raw if path]
        category_combined = '_'.join(categories_last)
        category_hash = hash(category_combined) % 100 / 100

        brand = item.get('brand', 'Unknown')
        brand_hash = hash(brand) % 100 / 100

        price = item.get('price', 0.0)
        if isinstance(price, str):
            price = price.replace('$', '').replace(',', '').strip()
        try:
            price = float(price)
            price_scaled = scaler.transform([[price]])[0][0]
        except ValueError:
            price_scaled = 0.0

        embedding = [category_hash, brand_hash, price_scaled]
        item_embeddings[item_id] = embedding

        if idx % 100000 == 0:
            print(f"已处理 {idx} 个商品，当前 embedding 数: {len(item_embeddings)}")

print(f"共生成 {len(item_embeddings)} 个 item embedding")

# === 第三步：整理用户历史行为 ===
print("整理用户历史行为 ...")
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
            print(f"已处理 {idx} 条 review，当前用户数: {len(user_histories)}")

for user_id in user_histories:
    user_histories[user_id].sort()

print(f"整理完成，共有 {len(user_histories)} 个用户历史，商品数: {len(all_items)}")

# === Note for Dataset Balancing ===
# If model training results show imbalance or poor performance on certain user groups:
# - Consider adjusting `user_sample_rate_high` and `user_sample_rate_low` to balance the presence of high/low interaction users.
# - You can also tune `neg_sample_num` to modify the positive/negative sample ratio for better ranking model effectiveness.

# === 第四步：构建训练样本并写入 CSV（流式处理 + 批量优化）===
# 参数配置
neg_sample_num = 2
flush_interval = 1000
user_sample_rate_high = 0.5  # 高质量用户采样率
user_sample_rate_low = 0.1   # 低交互用户采样率
min_history_len = 10         # 高质量用户最小交互数
max_history_len = 50
history_step = 2

print("构建优化后的训练样本并写入 train_data.csv ...")
train_data_path = os.path.join(base_dir, 'train_data.csv')

with open(train_data_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['user_id', 'history_items', 'candidate_item', 'label'])

    buffer_rows = []
    processed_users = 0

    for u_idx, (user_id, interactions) in enumerate(user_histories.items(), 1):
        interactions_sorted = sorted(interactions)
        interaction_count = len(interactions_sorted)

        # 高交互用户：交互 >= min_history_len
        if interaction_count >= min_history_len:
            if random.random() > user_sample_rate_high:
                continue
            start_idx = min_history_len  # 高交互从 min_history_len 开始
        else:
            # 低交互用户，部分保留
            if random.random() > user_sample_rate_low:
                continue
            start_idx = 1  # 低交互从第 1 条开始

        start_time = time.time()

        user_items = set([item for (_, item, _) in interactions_sorted])
        neg_items_pool = list(all_items - user_items)

        if len(neg_items_pool) > 5000:
            neg_items_pool = random.sample(neg_items_pool, 5000)

        if interaction_count > max_history_len:
            interactions_sorted = interactions_sorted[-max_history_len:]

        for idx in range(start_idx, len(interactions_sorted), history_step):
            history = [item for (_, item, _) in interactions_sorted[:idx]]
            if len(history) == 0:
                continue

            pos_item = interactions_sorted[idx][1]
            rating = interactions_sorted[idx][2]

            if rating >= 4:
                buffer_rows.append([user_id, '|'.join(history), pos_item, 1])

                if neg_items_pool:
                    neg_samples = random.sample(neg_items_pool, min(neg_sample_num, len(neg_items_pool)))
                    for neg_item in neg_samples:
                        buffer_rows.append([user_id, '|'.join(history), neg_item, 0])

            if len(buffer_rows) >= flush_interval:
                writer.writerows(buffer_rows)
                buffer_rows.clear()

        processed_users += 1

        if processed_users % 1000 == 0:
            print(f"已处理 {processed_users} 个用户，总用时 {time.time() - start_time:.2f}s")

    if buffer_rows:
        writer.writerows(buffer_rows)

print(f"训练样本写入完成！总共处理用户数: {processed_users}")

# === 第五步：保存 item_embeddings.csv ===
print("保存 item_embeddings.csv ...")
item_embedding_path = os.path.join(base_dir, 'item_embeddings.csv')
with open(item_embedding_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['item_id', 'category_hash', 'brand_hash', 'price_scaled'])  # 展开列名
    for item_id, embedding in item_embeddings.items():
        writer.writerow([item_id] + embedding)  # 直接写入浮点数列表