#!/bin/bash

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 检查是否有.env文件，如果没有则创建一个简单的.env文件
if [ ! -f "../.env" ]; then
  echo "创建默认.env文件..."
  echo "REDIS_HOST=localhost" > ../.env
  echo "REDIS_PORT=6379" >> ../.env
  echo "REDIS_DB=0" >> ../.env
fi

# 询问用户使用哪种启动方式
echo "请选择启动方式:"
echo "1) 连接到现有的Redis网络 (适用于Redis在Docker中运行)"
echo "2) 使用主机网络 (适用于Redis在本地运行或其他主机上)"
read -p "请输入选项 (默认: 2): " option

if [ "$option" = "1" ]; then
  # 询问Redis网络名
  read -p "请输入Redis的网络名 (默认: redis-network): " redis_network
  redis_network=${redis_network:-redis-network}
  
  # 检查网络是否存在，如果不存在则创建
  if ! docker network inspect "$redis_network" &>/dev/null; then
    echo "创建网络 $redis_network..."
    docker network create "$redis_network"
  fi
  
  # 启动服务
  echo "使用网络 $redis_network 启动模型服务..."
  docker compose -f model-compose.yml up --build -d
else
  # 启动使用主机网络的版本
  echo "使用主机网络启动模型服务..."
  docker compose -f model-compose-standalone.yml up --build -d
fi

echo "模型服务启动完成。可以通过以下命令查看日志:"
echo "docker logs model-service" 