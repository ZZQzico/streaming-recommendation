import json
import csv
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
books_path = os.path.join(base_dir, '../raw_data/Books_5.json')
excluded_users_path = os.path.join(base_dir, 'excluded_users.txt')
lightgcn_data_path = os.path.join(base_dir, 'lightgcn_data.csv')

# 加载 excluded_users.txt
print("加载 excluded_users.txt ...")
excluded_users = set()
with open(excluded_users_path, 'r') as f:
    for line in f:
        excluded_users.add(line.strip())
print(f"共加载 {len(excluded_users)} 个排除用户")

# 边读边处理 Books_5.json，生成 lightgcn_rows
print("开始处理 Books_5.json，生成 LightGCN 训练数据 ...")
lightgcn_rows = set()
with open(books_path, 'r') as f:
    for idx, line in enumerate(f, 1):
        review = json.loads(line)
        user_id = review['reviewerID']
        item_id = review['asin']
        rating = review['overall']

        if rating >= 4 and user_id not in excluded_users:
            lightgcn_rows.add((user_id, item_id))

        if idx % 1000000 == 0:
            print(f"已处理 {idx} 条 review，当前 LightGCN 样本数: {len(lightgcn_rows)}")

print(f"Books_5.json 处理完成，共生成 {len(lightgcn_rows)} 条 LightGCN 样本")

# 保存 lightgcn_data.csv
print(f"保存 {lightgcn_data_path} ...")
with open(lightgcn_data_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['user_id', 'item_id'])
    for idx, (user_id, item_id) in enumerate(lightgcn_rows, 1):
        writer.writerow([user_id, item_id])
        if idx % 1000000 == 0:
            print(f"已写入 {idx} 条 LightGCN 样本 ...")

print("LightGCN 数据生成完成！")