# 电商客服 Agent · 部署镜像（CPU）
FROM python:3.12-slim

WORKDIR /app

# 先装 CPU 版 torch（避免默认拉取 2.5GB+ 的 CUDA 版）
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# 构建时注入：HF_ENDPOINT 便于国内拉取 embedding 模型
ENV HF_ENDPOINT=https://hf-mirror.com

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
