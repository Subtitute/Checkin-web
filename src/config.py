# File: src/config.py
import os

class Config:
    # --- Đường dẫn dữ liệu ---
    data_dir = "data"
    raw_image_dir = os.path.join(data_dir, "raw")
    face_dir = os.path.join(data_dir, "faces")
    processed_dir = os.path.join(data_dir, "processed")
    
    # --- File dữ liệu ---
    csv_info_file = "data.csv"
    guest_csv_file = "guests.csv"
    embeddings_file = "face_embeddings.npy"
    checkin_log = os.path.join(processed_dir, "check_log.csv")
    
    # --- CẤU HÌNH CAMERA ---
    CAMERA_INDEX = 0
    CAMERA_URL = "rtsp://admin:EcoVision1@192.168.0.64:554/Streaming/Channels/102"

    # --- CẤU HÌNH TELEGRAM (Bắt buộc phải điền để hết cảnh báo) ---
    # Dán Token và Chat ID bạn vừa lấy vào đây
    TELEGRAM_BOT_TOKEN = "8670125979:AAEJcGkl1vNUuwRpk1LAkNK-AkaKGAKCROQ"  # Thay bằng token thật
    TELEGRAM_CHAT_ID = "6706593812"  # Thay bằng chat_id thật
    
    # --- CẤU HÌNH GIỜ LÀM VIỆC (ĐANG THIẾU ĐOẠN NÀY - HÃY THÊM VÀO) ---
    WORK_START_TIME = "07:00"   # Giờ bắt đầu ca sáng
    WORK_END_TIME = "17:30"     # Giờ kết thúc ca chiều
    
    # --- Cấu hình Model AI ---
    model_name = "buffalo_l"
    ctx_id = -1  # -1 = CPU, >0 = GPU ID
    det_size = (640, 640)
    
    # --- Tham số xử lý ảnh ---
    image_max_size = 1600
    crop_margin_ratio = 0.6
    output_face_size = (512, 512)
    min_face_size = 60
    
    # --- Tham số điểm danh ---
    confirm_frames = 3
    checkin_cooldown = 5
    threshold = 0.4
    
    show_fps = True