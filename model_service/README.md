# 推荐系统模型服务

这是一个基于FastAPI的推荐系统模型服务，它集成了三个模型（LightGCN、DIN和RankNet）并以REST API的形式提供推荐功能。

## 功能特点

- 从Redis获取用户历史行为数据
- 使用多个推荐模型生成个性化推荐结果
- 将推荐结果写回Redis，供前端使用
- 提供REST API端点供直接调用
- 支持持续监听Redis中的用户数据
- 提供健康检查和性能指标API

## 系统要求

- Python 3.9+
- PyTorch
- FastAPI 
- Redis
- Docker (可选，推荐使用)

## 快速开始

### 使用脚本启动（推荐）

我们提供了一个启动脚本，能够根据你的环境自动选择合适的配置：

```bash
# 进入模型服务目录
cd model_service

# 执行启动脚本（可能需要先添加执行权限）
chmod +x start_model_service.sh
./start_model_service.sh
```

脚本会引导你选择合适的启动方式：
1. 连接到现有的Redis Docker网络（适用于Redis也运行在Docker中的情况）
2. 使用主机网络（适用于Redis运行在本地或远程主机上的情况）

### 使用Docker Compose（集成到现有系统）

如果你想将模型服务集成到现有的Docker环境中：

```bash
# 进入模型服务目录
cd model_service

# 使用model-compose.yml启动
docker compose -f model-compose.yml up -d
```

### 使用Docker Compose（独立运行）

如果你想独立运行模型服务，并连接到本地或远程的Redis：

```bash
# 设置Redis地址（可选，默认为localhost）
export REDIS_HOST=你的Redis地址
export REDIS_PORT=6379
export REDIS_DB=0

# 进入模型服务目录
cd model_service

# 启动服务
docker compose -f model-compose-standalone.yml up -d
```

### 使用完整的docker-compose（包含所有服务）

如果你想启动包含Kafka、Redis和所有其他服务的完整环境：

```bash
# 在项目根目录下
docker-compose build model-service
docker-compose up -d model-service
```

### 手动运行

1. 安装依赖:

```bash
pip install torch pandas numpy redis fastapi uvicorn python-dotenv
```

2. 确保`.env`文件包含必要的配置项:

```
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0
```

3. 启动服务:

```bash
cd model_service
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

## 测试服务

### 健康检查

```bash
curl http://localhost:8001/health
```

正常响应：
```json
{"status":"healthy","model_loaded":true}
```

### 获取指标

```bash
curl http://localhost:8001/metrics
```

### 手动触发推荐生成

为特定用户触发推荐:

```bash
curl -X 'POST' 'http://localhost:8001/recommend/{user_id}' -H 'accept: application/json'
```

例如:

```bash
curl -X 'POST' 'http://localhost:8001/recommend/A2SRVEKZS33OGN' -H 'accept: application/json'
```

### 测试完整流程

1. 向Redis写入用户历史数据:

```bash
docker exec -it redis redis-cli SET "user_profile:A2SRVEKZS33OGN" '["0764227971", "0990351408", "1492635227", "0099740915", "0764213466"]'
```

2. 触发推荐生成:

```bash
curl -X 'POST' 'http://localhost:8001/recommend/A2SRVEKZS33OGN' -H 'accept: application/json'
```

3. 从Redis获取推荐结果:

```bash
docker exec -it redis redis-cli GET "recommendations:A2SRVEKZS33OGN"
```

## API端点

- `GET /health`: 健康检查
- `GET /metrics`: 服务指标
- `POST /recommend/{user_id}`: 为指定用户生成推荐

## 故障排查

1. 如果服务无法连接到Redis:
   - 检查Redis服务是否运行
   - 验证环境变量中的Redis配置是否正确
   - 如果使用Docker网络，确保模型服务和Redis在同一网络中

2. 如果模型加载失败:
   - 确保模型文件位于正确的路径(`/app/model_output/[model_name]/`)
   - 检查模型文件格式是否正确

3. 如果推荐结果为空:
   - 检查用户历史数据是否存在于Redis中
   - 验证用户ID格式是否正确

## 日志监控

查看模型服务的日志:

```bash
docker logs model-service
```

## 架构说明

该服务采用多阶段推荐架构:
1. 使用LightGCN模型进行协同过滤，生成广泛候选集
2. 使用DIN模型对候选集进行粗排序
3. 使用RankNet模型对排序结果进行精排，生成最终推荐列表 