# File: src/liveness_check.py
import os
import cv2
import numpy as np
import torch
import torch.nn as nn

# --- KHAI BÁO BIẾN TOÀN CỤC CỨNG (HARD-CODED) ---
# Khởi tạo ngay là False. Tuyệt đối không gán từ biến khác hay đọc từ file.
MODEL_AVAILABLE = False
predictor_model = None

# Xác định thiết bị chạy (CPU hoặc GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- ĐỊNH NGHĨA KIẾN TRÚC MẠNG (GIỐNG HỆT LÚC TRAIN) ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.25),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 10 * 10, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def _load_model_internal():
    """
    Hàm nội bộ để nạp model. 
    Được gọi duy nhất một lần khi khởi động script.
    """
    global MODEL_AVAILABLE, predictor_model
    
    # Reset về trạng thái an toàn mặc định
    MODEL_AVAILABLE = False
    predictor_model = None
    
    model_path = "models/liveness_model_robust_v2.pth"
    
    # Kiểm tra file tồn tại
    if not os.path.exists(model_path):
        print(f"[WARN] File model không tồn tại: {model_path}")
        print("[INFO] Hệ thống sẽ bỏ qua bước kiểm tra Liveness.")
        return

    try:
        print(f"[INFO] Đang nạp model Liveness từ: {model_path}...")
        
        # Khởi tạo kiến trúc
        predictor_model = SimpleCNN().to(device)
        
        # Nạp trọng số
        checkpoint = torch.load(model_path, map_location=device)
        
        # Xử lý linh hoạt định dạng checkpoint
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('state_dict', checkpoint)
            predictor_model.load_state_dict(state_dict)
        else:
            predictor_model.load_state_dict(checkpoint)
            
        predictor_model.eval()
        
        # CẬP NHẬT TRẠNG THÁI THÀNH CÔNG (BOOLEAN TRUE)
        MODEL_AVAILABLE = True
        print(f"[SUCCESS] Model Liveness đã sẵn sàng trên thiết bị: {device}")
        
    except Exception as e:
        print(f"[ERROR] Lỗi nghiêm trọng khi nạp model: {e}")
        MODEL_AVAILABLE = False
        predictor_model = None

# Gọi hàm nạp model ngay khi file được import
_load_model_internal()

def check_liveness(face_crop_img):
    """
    Hàm kiểm tra thật/giả.
    Input: Ảnh crop khuôn mặt (BGR).
    Output: (is_real: bool, score: float)
    """
    # KIỂM TRA AN TOÀN: Chỉ kiểm tra biến boolean MODEL_AVAILABLE
    if not MODEL_AVAILABLE or predictor_model is None:
        # Fail-open: Nếu không có model, coi như là thật để không chặn người dùng
        return True, 1.0

    try:
        # Tiền xử lý ảnh chuẩn hóa
        img_resized = cv2.resize(face_crop_img, (80, 80))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype("float32") / 255.0
        
        # Chuyển sang Tensor
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = predictor_model(img_tensor)
            prob = output.item()
            
        # Ngưỡng quyết định
        threshold = 0.45
        is_real = (prob > threshold)
        
        return is_real, float(prob)
        
    except Exception as e:
        print(f"[ERROR] Lỗi suy luận Liveness: {e}")
        return False, 0.0