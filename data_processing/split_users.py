import json
import random
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, '../raw_data/Books_5.json')

# 获取全部用户（边读边处理）
print("开始提取用户ID ...")
user_set = set()
with open(file_path, 'r') as f:
    for idx, line in enumerate(f, 1):
        review = json.loads(line)
        user_set.add(review['reviewerID'])
        if idx % 1000000 == 0:
            print(f"已处理 {idx} 条 review，当前用户数: {len(user_set)}")

print(f"用户ID提取完成，共有 {len(user_set)} 个不同用户")

# 打乱用户列表
user_list = list(user_set)
random.shuffle(user_list)

# 划分 80% vs 20%
split_index = int(0.8 * len(user_list))
train_users = set(user_list[:split_index])
excluded_users = set(user_list[split_index:])

print(f"训练用户数量: {len(train_users)}")
print(f"排除用户数量: {len(excluded_users)}")

# 保存 excluded_users.txt
excluded_users_path = os.path.join(base_dir, 'excluded_users.txt')
with open(excluded_users_path, 'w') as f:
    for user in excluded_users:
        f.write(user + '\n')

print(f"excluded_users 已保存至 {excluded_users_path}")