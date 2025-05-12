import time
import random
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置
FASTAPI_URL = "http://localhost:8000"
ENDPOINT = "/predict/"
USER_IDS_FILE = "data/user_ids.txt"
CONCURRENCY = 20
TOTAL_REQUESTS = 500

# 加载用户ID列表
with open(USER_IDS_FILE) as f:
    user_ids = [line.strip() for line in f if line.strip()]


def single_request(user_id):
    ts = int(time.time())
    start = time.time()
    try:
        r = requests.post(
            FASTAPI_URL + ENDPOINT,
            json={"user_id": user_id, "timestamp": ts},
            timeout=5
        )
        latency = time.time() - start
        return latency, r.status_code
    except Exception:
        return None, None


def main():
    latencies = []
    errors = 0
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = [executor.submit(single_request, random.choice(user_ids))
                   for _ in range(TOTAL_REQUESTS)]
        for f in as_completed(futures):
            lat, status = f.result()
            if lat is None or status is None:
                errors += 1
            else:
                latencies.append(lat)
                if status >= 400:
                    errors += 1

    avg_lat = sum(latencies) / len(latencies) if latencies else float('nan')
    print(f"Total: {TOTAL_REQUESTS}, Success: {len(latencies)}, Errors: {errors}")
    print(f"Avg latency: {avg_lat:.3f}s, P95 latency: {sorted(latencies)[int(len(latencies)*0.95)]:.3f}s")


if __name__ == "__main__":
    main()
