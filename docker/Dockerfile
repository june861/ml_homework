# 使用官方的 Python 镜像作为基础镜像
FROM python:3.10-slim

ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
# 设置工作目录
WORKDIR /project

# 复制当前目录的所有文件到容器内的工作目录
COPY ../.. /project

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 暴露容器的端口（例如 Flask 应用通常监听 5000 端口）
EXPOSE 5000

# 设置容器启动时执行的命令
CMD ["python", "app.py"]
