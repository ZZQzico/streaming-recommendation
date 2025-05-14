#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import time
import random
import json
import argparse
import datetime
import os
import csv
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any

# 配置常量
API_ENDPOINT = "http://localhost:8000"
MODEL_ENDPOINT = "http://localhost:8001"
# 推荐生成等待时间（秒）
REC_WAIT_TIME = 5

# 从CSV文件加载实际用户ID
def load_user_ids(csv_path='data_processing/excluded_users_behavior.csv'):
    user_ids = []
    files_to_try = [
        csv_path,
        f"../{csv_path}",
        "excluded_users_behavior.csv",
        "data_processing/excluded_users_behavior.csv", 
        "../data_processing/excluded_users_behavior.csv"
    ]
    
    for file_path in files_to_try:
        try:
            if os.path.exists(file_path):
                print(f"尝试从 {file_path} 读取用户数据...")
                with open(file_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if 'user_id' in row and row['user_id']:
                            user_ids.append(row['user_id'])
                        # 如果没有user_id列名，尝试取第一列
                        elif row and list(row.values())[0]:
                            user_ids.append(list(row.values())[0])
                
                if user_ids:
                    print(f"从 {file_path} 成功加载 {len(user_ids)} 个用户ID")
                    # 取一小部分用户ID做测试，避免列表太大
                    if len(user_ids) > 20:
                        user_ids = random.sample(user_ids, 20)
                        print(f"随机选择 20 个用户ID进行测试")
                    return user_ids
        except Exception as e:
            print(f"读取 {file_path} 时出错: {str(e)}")
    
    # 如果所有文件都无法读取，使用默认值
    print("警告: 无法找到或读取用户数据文件，使用默认用户ID")
    return ["user1", "user2", "user3", "user4", "user5"]

# 检查哪些用户能提供有效的推荐结果
def find_working_users(user_ids, verbose=True):
    working_users = []
    if verbose:
        print("\n尝试查找能提供有效推荐的用户...")
    
    # 尝试使用固定的已知工作时间戳
    timestamp = 1672560000  # 2023-01-01 00:00:00
    
    for i, user_id in enumerate(user_ids[:10]):  # 只检查前10个用户
        try:
            if verbose:
                print(f"测试用户 {user_id} 是否能获得推荐结果...", end="")
                
            # 发送请求
            url = f"{API_ENDPOINT}/push_interest/"
            response = requests.post(url, json={"user_id": user_id, "timestamp": timestamp})
            
            if response.status_code == 200:
                # 等待推荐系统处理
                time.sleep(REC_WAIT_TIME)  
                
                # 获取推荐结果
                rec_url = f"{MODEL_ENDPOINT}/recommendations/{user_id}"
                rec_response = requests.get(rec_url)
                
                if rec_response.status_code == 200:
                    rec_data = rec_response.json()
                    num_recs = len(rec_data.get("recommendations", []))
                    
                    if num_recs > 0:
                        working_users.append(user_id)
                        if verbose:
                            print(f" ✓ 成功! 获得 {num_recs} 个推荐结果")
                    else:
                        if verbose:
                            print(f" ⚠ 无推荐结果 (成功请求但返回空列表)")
                else:
                    if verbose:
                        print(f" ✗ 获取推荐失败: {rec_response.status_code}")
            else:
                if verbose:
                    print(f" ✗ 发送请求失败: {response.status_code}")
                    
        except Exception as e:
            if verbose:
                print(f" ✗ 错误: {str(e)}")
    
    if working_users:
        if verbose:
            print(f"\n找到 {len(working_users)} 个能提供推荐的用户: {working_users}")
        return working_users
    else:
        if verbose:
            print("\n未找到能提供推荐的用户，使用原始用户列表")
        return user_ids

# 加载实际用户ID
SAMPLE_USERS = load_user_ids()
# 找到能提供推荐的用户
WORKING_USERS = find_working_users(SAMPLE_USERS)

# 测试统计数据
class TestStats:
    def __init__(self):
        self.requests_sent = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.start_time = time.time()
        self.latencies = []
        self.recommendation_counts = []
    
    def add_request(self, success: bool, latency: float):
        self.requests_sent += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        self.latencies.append(latency)
    
    def add_recommendation_count(self, count: int):
        self.recommendation_counts.append(count)
        
    def print_summary(self):
        duration = time.time() - self.start_time
        print("\n===== 测试结果摘要 =====")
        print(f"总运行时间: {duration:.2f} 秒")
        print(f"总请求数: {self.requests_sent}")
        print(f"成功请求: {self.successful_requests}")
        print(f"失败请求: {self.failed_requests}")
        
        if self.latencies:
            avg_latency = sum(self.latencies) / len(self.latencies)
            max_latency = max(self.latencies)
            min_latency = min(self.latencies)
            print(f"平均延迟: {avg_latency*1000:.2f} ms")
            print(f"最大延迟: {max_latency*1000:.2f} ms")
            print(f"最小延迟: {min_latency*1000:.2f} ms")
            print(f"每秒请求数: {self.requests_sent / duration:.2f}")
            
        if self.recommendation_counts:
            avg_recs = sum(self.recommendation_counts) / len(self.recommendation_counts)
            print(f"平均推荐数量: {avg_recs:.2f}")
            print(f"最大推荐数量: {max(self.recommendation_counts)}")
            print(f"总计获得推荐: {sum(self.recommendation_counts)}")
            print(f"获得推荐的请求比例: {sum(1 for c in self.recommendation_counts if c > 0) / len(self.recommendation_counts):.2%}")
        
        print("========================")

# 发送推荐请求和获取推荐结果
def send_request(stats: TestStats, endpoint: str, user_id: str, timestamp: int) -> Dict[str, Any]:
    try:
        start_time = time.time()
        if endpoint == "push_interest":
            url = f"{API_ENDPOINT}/push_interest/"
            payload = {"user_id": user_id, "timestamp": timestamp}
            response = requests.post(url, json=payload)
        elif endpoint == "send_kafka":
            url = f"{API_ENDPOINT}/send_kafka/"
            payload = {"user_id": user_id, "timestamp": timestamp}
            response = requests.post(url, json=payload)
        elif endpoint == "get_recommendations":
            url = f"{MODEL_ENDPOINT}/recommendations/{user_id}"
            response = requests.get(url)
        else:
            print(f"Unknown endpoint: {endpoint}")
            return {"success": False, "error": "Unknown endpoint"}
        
        latency = time.time() - start_time
        stats.add_request(response.status_code == 200, latency)
        
        if response.status_code == 200:
            response_data = response.json()
            return {
                "success": True,
                "user_id": user_id,
                "endpoint": endpoint,
                "status_code": response.status_code,
                "latency": latency,
                "response": response_data
            }
        else:
            return {
                "success": False,
                "user_id": user_id,
                "endpoint": endpoint,
                "status_code": response.status_code,
                "latency": latency,
                "error": response.text
            }
    except Exception as e:
        print(f"Error sending request to {endpoint}: {str(e)}")
        stats.add_request(False, 0)
        return {"success": False, "user_id": user_id, "endpoint": endpoint, "error": str(e)}

# 生成可能有用的时间戳
def generate_timestamp() -> int:
    # 基础日期时间戳（2023年各月）
    base_dates = [
        datetime.datetime(2023, 1, 1),
        datetime.datetime(2023, 2, 1),
        datetime.datetime(2023, 3, 1),
        datetime.datetime(2023, 4, 1),
        datetime.datetime(2023, 5, 1),
        datetime.datetime(2023, 6, 1),
        datetime.datetime(2023, 7, 1),
        datetime.datetime(2023, 8, 1),
        datetime.datetime(2023, 9, 1),
        datetime.datetime(2023, 10, 1),
        datetime.datetime(2023, 11, 1),
        datetime.datetime(2023, 12, 1)
    ]
    
    # 随机选择一个基础日期
    base_date = random.choice(base_dates)
    
    # 为日期添加随机小时和分钟
    hour = random.randint(0, 23)
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    
    # 在基础日期上增加随机天数（0-6天，保持在当月范围内）
    random_days = random.randint(0, 6)
    
    # 构建最终日期时间
    timestamp_date = base_date + datetime.timedelta(days=random_days, 
                                                   hours=hour, 
                                                   minutes=minute,
                                                   seconds=second)
    
    # 转换为Unix时间戳
    return int(timestamp_date.timestamp())

# 执行测试场景
def run_test(num_requests: int, concurrent: int, delay: float, verbose: bool):
    stats = TestStats()
    print(f"开始执行测试: {num_requests} 请求, 并发数 {concurrent}, 延迟 {delay}s")
    print(f"推荐生成等待时间: {REC_WAIT_TIME}秒")
    if WORKING_USERS != SAMPLE_USERS:
        print(f"使用有效推荐用户: {WORKING_USERS[:3]} ... (共 {len(WORKING_USERS)} 个)")
    else:
        print(f"使用采样用户: {SAMPLE_USERS[:3]} ... (共 {len(SAMPLE_USERS)} 个)")
    
    def process_request():
        # 优先使用已知能工作的用户
        user_id = random.choice(WORKING_USERS if WORKING_USERS else SAMPLE_USERS)
        timestamp = generate_timestamp()
        
        # 随机选择端点:
        endpoint = random.choice(["push_interest", "send_kafka"])
        result = send_request(stats, endpoint, user_id, timestamp)
        
        if verbose:
            if result["success"]:
                print(f"✓ {endpoint} - 用户: {user_id}, 延迟: {result['latency']*1000:.2f}ms")
            else:
                print(f"✗ {endpoint} - 用户: {user_id}, 错误: {result.get('error', 'Unknown error')}")
        
        # 如果成功发送了推荐请求，尝试获取推荐结果
        if result["success"] and endpoint in ["push_interest", "send_kafka"]:
            # 等待推荐系统处理
            if verbose:
                print(f"  等待 {REC_WAIT_TIME} 秒让推荐系统处理数据...")
            time.sleep(REC_WAIT_TIME)  
            
            rec_result = send_request(stats, "get_recommendations", user_id, timestamp)
            if rec_result["success"]:
                num_recs = len(rec_result["response"].get("recommendations", []))
                stats.add_recommendation_count(num_recs)
                if verbose:
                    if num_recs > 0:
                        print(f"  → 获取到 {num_recs} 个推荐结果, 延迟: {rec_result['latency']*1000:.2f}ms")
                    else:
                        print(f"  → 无推荐结果, 延迟: {rec_result['latency']*1000:.2f}ms")
            else:
                stats.add_recommendation_count(0)
                if verbose:
                    print(f"  → 获取推荐失败: {rec_result.get('error', 'Unknown error')}")
        
        time.sleep(delay)  # 请求间隔
    
    # 顺序执行或并发执行
    if concurrent <= 1:
        for _ in range(num_requests):
            process_request()
    else:
        with ThreadPoolExecutor(max_workers=concurrent) as executor:
            for _ in range(num_requests):
                executor.submit(process_request)
    
    stats.print_summary()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="推荐系统负载测试工具")
    parser.add_argument("-n", "--num-requests", type=int, default=10,
                        help="要发送的请求数量 (默认: 10)")
    parser.add_argument("-c", "--concurrent", type=int, default=1,
                        help="并发请求数 (默认: 1)")
    parser.add_argument("-d", "--delay", type=float, default=1.0,
                        help="请求之间的延迟，秒 (默认: 1.0)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="显示详细输出")
    parser.add_argument("-w", "--wait-time", type=float, default=5.0,
                        help="获取推荐前的等待时间，秒 (默认: 5.0)")
    
    args = parser.parse_args()
    
    # 设置推荐等待时间
    REC_WAIT_TIME = args.wait_time
    
    try:
        print("推荐系统负载测试")
        print("-------------------")
        run_test(args.num_requests, args.concurrent, args.delay, args.verbose)
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n测试出错: {str(e)}") 