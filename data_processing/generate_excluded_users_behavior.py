import json
import os

# 路径配置
base_dir = os.path.dirname(os.path.abspath(__file__))
raw_data_path = os.path.join(base_dir, '../raw_data/Books_5.json')
excluded_users_path = os.path.join(base_dir, 'excluded_users.txt')
output_behavior_path = os.path.join(base_dir, 'excluded_users_behavior.csv')

# 读excluded users
print("加载 excluded_users.txt ...")
excluded_users = set()
with open(excluded_users_path, 'r') as f:
    for line in f:
        excluded_users.add(line.strip())

print(f"排除用户数量: {len(excluded_users)}")

# 处理Books_5.json，提取行为
print("开始提取 excluded users 的行为 ...")
behaviors = []

with open(raw_data_path, 'r') as f:
    for idx, line in enumerate(f, 1):
        review = json.loads(line)
        user_id = review['reviewerID']
        item_id = review['asin']
        timestamp = review['unixReviewTime']  # 时间戳是整数秒

        if user_id in excluded_users:
            behaviors.append((user_id, item_id, timestamp))

        if idx % 1000000 == 0:
            print(f"已处理 {idx} 条 review")

# 保存到CSV
print(f"总共提取到 {len(behaviors)} 条 excluded user 行为")
behaviors.sort(key=lambda x: (x[0], x[2]))  # 按 user_id + timestamp 排序

with open(output_behavior_path, 'w') as f:
    f.write('user_id,item_id,timestamp\n')
    for user_id, item_id, timestamp in behaviors:
        f.write(f"{user_id},{item_id},{timestamp}\n")

print(f"excluded_users_behavior.csv 已保存至 {output_behavior_path}")
