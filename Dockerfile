# Sử dụng ảnh nền Python 3.10 slim
FROM python:3.10-slim

# Thiết lập biến môi trường để tránh các tương tác hỏi đáp khi cài đặt
ENV DEBIAN_FRONTEND=noninteractive

# Cài đặt các thư viện hệ thống cần thiết
# QUAN TRỌNG: Thêm 'build-essential' để cài g++ cho insightface
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libimagequant0 \
    ffmpeg \
    curl \
    git \
    cmake \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Thiết lập thư mục làm việc
WORKDIR /app

# Copy requirements và cài đặt
COPY requirements.txt .

# Nâng cấp pip trước khi cài để tránh lỗi tương thích
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code vào container
# Lưu ý: Các file trong .dockerignore sẽ bị bỏ qua
COPY . .

# Tạo các thư mục dữ liệu rỗng nếu chưa có
RUN mkdir -p data/raw data/faces data/processed models

# Biến môi trường cho cấu hình ứng dụng
# Lưu ý: Không nên hard-code token nhạy cảm vào Dockerfile. 
# Hãy truyền chúng khi chạy docker run bằng flag -e
ENV TELEGRAM_BOT_TOKEN="8670125979:AAEJcGkl1vNUuwRpk1LAkNK-AkaKGAKCROQ"
ENV TELEGRAM_CHAT_ID="6706593812"
ENV CAMERA_INDEX="0"
ENV CAMERA_URL=""

# Cổng mặc định nếu bạn có chạy Flask API
EXPOSE 5000

# Lệnh chạy ứng dụng
CMD ["python", "main.py"]