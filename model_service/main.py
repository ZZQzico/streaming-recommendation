import os
import json
import logging
import asyncio
from fastapi import FastAPI, BackgroundTasks
from dotenv import load_dotenv
import redis
from models import RecommendationModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)

# 加载环境变量
load_dotenv()

# Redis配置
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

# 创建FastAPI应用
app = FastAPI(title="推荐系统模型服务")

# 创建Redis连接
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    decode_responses=True
)

# 加载推荐模型
recommendation_model = RecommendationModel()

# 处理单个用户的后台任务
async def process_user(user_id):
    # 从Redis获取用户历史行为
    user_key = f"user_profile:{user_id}"
    user_history = redis_client.get(user_key)
    
    if not user_history:
        logging.warning(f"未找到用户 {user_id} 的历史数据")
        return
    
    # 解析历史数据
    try:
        history_items = json.loads(user_history)
        
        # 使用模型预测推荐结果
        recommendations = recommendation_model.predict(user_id, history_items)
        
        # 将推荐结果存回Redis
        result_key = f"recommendations:{user_id}"
        redis_client.set(result_key, json.dumps(recommendations))
        
        logging.info(f"用户 {user_id} 的推荐结果已存入Redis, 共 {len(recommendations)} 个物品")
    except Exception as e:
        logging.error(f"处理用户 {user_id} 时出错: {str(e)}")

# 持续监听用户数据的后台任务
async def continuous_monitoring():
    logging.info("启动连续监听任务...")
    while True:
        try:
            # 获取所有用户资料键
            user_profile_keys = redis_client.keys("user_profile:*")
            
            if not user_profile_keys:
                await asyncio.sleep(1)
                continue
            
            for key in user_profile_keys:
                user_id = key.split(":")[-1]
                await process_user(user_id)
                
            # 等待一段时间后再次检查
            await asyncio.sleep(1)
        except Exception as e:
            logging.error(f"连续监听任务出错: {str(e)}")
            await asyncio.sleep(5)  # 出错后等待时间更长

# 启动事件：启动后台任务
@app.on_event("startup")
async def startup_event():
    # 启动持续监听任务
    app.state.background_task = asyncio.create_task(continuous_monitoring())
    logging.info("模型服务已启动，正在监听Redis中的用户数据...")

# 关闭事件：清理资源
@app.on_event("shutdown")
async def shutdown_event():
    # 取消后台任务
    if hasattr(app.state, "background_task"):
        app.state.background_task.cancel()
        
    logging.info("模型服务正在关闭...")

# 手动触发对特定用户的推荐
@app.post("/recommend/{user_id}")
async def recommend(user_id: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_user, user_id)
    return {"status": "success", "message": f"已开始为用户 {user_id} 生成推荐"}

# 健康检查端点
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": recommendation_model.is_ready()}

# 性能指标端点
@app.get("/metrics")
async def get_metrics():
    return {
        "redis_keys": len(redis_client.keys("*")),
        "user_profiles": len(redis_client.keys("user_profile:*")),
        "recommendations": len(redis_client.keys("recommendations:*"))
    } 