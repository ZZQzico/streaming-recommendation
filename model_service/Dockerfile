FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装依赖
RUN pip install --no-cache-dir torch pandas numpy redis fastapi uvicorn python-dotenv

# 复制模型文件和代码
COPY ./model_output /app/model_output
COPY ./model_service/main.py /app/model_service/
COPY ./model_service/models.py /app/model_service/models.py
COPY ./.env /app

# 设置工作目录
WORKDIR /app/model_service

# 运行FastAPI服务
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"] 