import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- CẤU HÌNH ---
DATASET_ROOT = "datasets"
CSV_REPLAY_PATH = os.path.join(DATASET_ROOT, "Replay-Attack", "replay_30.csv")
MODEL_SAVE_PATH = "models/liveness_model_robust_v2.pth"
LOG_EXCEL_PATH = "data/processed/train_log_robust_v2.xlsx"
PLOT_SAVE_PATH = "data/processed/train_evaluation_robust_v2.png"

IMG_SIZE = (80, 80)
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.0005

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Đang sử dụng thiết bị: {device}")

# --- DATA AUGMENTATION MẠNH MẼ CHỐNG ÁNH SÁNG (ĐÃ SỬA LỖI) ---
train_transform = A.Compose([
    # Sửa lỗi Brightness/Contrast
    A.RandomBrightnessContrast(brightness_limit=0.6, contrast_limit=0.6, p=0.8),
    
    # Sửa lỗi màu sắc
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=30, p=0.5),
    
    # Sửa lỗi GaussNoise: thay var_limit bằng sigma_limit
    A.GaussNoise(sigma_limit=(10.0, 50.0), p=0.5),
    
    # Sửa lỗi ImageCompression: thay quality_lower/upper bằng quality_range
    A.ImageCompression(quality_range=(60, 90), p=0.4),
    
    # Sửa lỗi GammaCorrection: thay bằng RandomGamma
    A.RandomGamma(gamma_limit=(60, 150), p=0.5),
    
    # Các phép biến đổi hình học
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    
    A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


class LivenessDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=img_rgb)
            img_tensor = augmented['image']
        else:
            img_norm = img_rgb.astype("float32") / 255.0
            img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1)
            
        return img_tensor, torch.tensor(label, dtype=torch.float32)

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

def extract_frames_from_video(video_path, output_dir, label_name, max_frames=5):
    """Cắt video thành các frame và lưu vào output_dir"""
    if not os.path.exists(video_path):
        return 0
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return 0
    
    # Lấy các frame cách đều nhau
    indices = [int(total_frames * i / (max_frames + 1)) for i in range(1, max_frames + 1)]
    count = 0
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Lưu ảnh với tên duy nhất
            fname = f"{os.path.basename(os.path.dirname(video_path))}_frame_{idx}_{label_name}.jpg"
            cv2.imwrite(os.path.join(output_dir, fname), frame)
            count += 1
    cap.release()
    return count

def prepare_replay_attack():
    """
    Xử lý dataset Replay-Attack dựa trên file replay_30.csv.
    Tự động trích xuất cả Fake (từ link) và Real (từ live_video_id).
    """
    if not os.path.exists(CSV_REPLAY_PATH):
        print("[WARN] Không tìm thấy file replay_30.csv. Bỏ qua bước tự động xử lý Replay-Attack.")
        return

    print("[INFO] Đang xử lý dataset Replay-Attack dựa trên file CSV...")
    try:
        df = pd.read_csv(CSV_REPLAY_PATH)
    except Exception as e:
        print(f"[ERROR] Lỗi đọc file CSV: {e}")
        return
    
    samples_dir = os.path.join(DATASET_ROOT, "Replay-Attack", "samples")
    
    # Thư mục đích
    real_out_dir = os.path.join(DATASET_ROOT, "REPLAY_REAL")
    fake_out_dir = os.path.join(DATASET_ROOT, "REPLAY_FAKE")
    
    os.makedirs(real_out_dir, exist_ok=True)
    os.makedirs(fake_out_dir, exist_ok=True)
    
    fake_count = 0
    real_count = 0
    missing_real_ids = []

    # --- 1. XỬ LÝ DỮ LIỆU FAKE (Từ cột 'link') ---
    print("   - Đang trích xuất ảnh FAKE...")
    for idx, row in df.iterrows():
        link = row['link']
        if pd.isna(link): continue
        
        # Link format: "folder_id/replay_video.mp4"
        folder_id = str(link).split('/')[0]
        video_file = os.path.join(samples_dir, folder_id, "replay_video.mp4")
        
        if os.path.exists(video_file):
            count = extract_frames_from_video(video_file, fake_out_dir, "FAKE")
            fake_count += count
        else:
            # Thử tìm file .mov hoặc tên khác nếu không phải .mp4
            # (Tùy biến thể dataset, nhưng thường là replay_video.mp4)
            pass

    # --- 2. XỬ LÝ DỮ LIỆU REAL (Từ cột 'live_video_id') ---
    print("   - Đang tìm kiếm và trích xuất ảnh REAL...")
    
    # Tạo danh sách tất cả folder có sẵn trong samples để tra cứu nhanh
    available_folders = set(os.listdir(samples_dir)) if os.path.exists(samples_dir) else set()
    
    # Danh sách các ID thực tế cần tìm
    unique_real_ids = df['live_video_id'].dropna().unique()
    
    for real_id in unique_real_ids:
        real_id = str(real_id).strip()
        found = False
        
        # Trường hợp 1: Folder thật nằm ngay trong samples (hiếm gặp nhưng kiểm tra cho chắc)
        if real_id in available_folders:
            video_path = os.path.join(samples_dir, real_id, "live_video.mp4") # Hoặc tên file gốc
            # Thường file thật trong dataset gốc có thể tên khác, ví dụ: video_id.mp4
            # Ta cần quét xem trong folder đó có file video nào không
            files_in_folder = [f for f in os.listdir(os.path.join(samples_dir, real_id)) if f.endswith(('.mp4', '.avi', '.mov')) and 'replay' not in f.lower()]
            if files_in_folder:
                video_path = os.path.join(samples_dir, real_id, files_in_folder[0])
                count = extract_frames_from_video(video_path, real_out_dir, "REAL")
                real_count += count
                found = True

        # Trường hợp 2: Video thật nằm ở folder khác (ví dụ: datasets/Replay-Attack/real/)
        # Nếu dataset của bạn chỉ có folder 'samples' chứa attack, thì khả năng cao video gốc 
        # chưa được giải nén hoặc nằm ở nơi khác.
        # Code sẽ ghi nhận ID bị thiếu để bạn biết.
        
        if not found:
            missing_real_ids.append(real_id)

    print(f"[INFO] Đã trích xuất {fake_count} ảnh FAKE từ Replay-Attack.")
    print(f"[INFO] Đã trích xuất {real_count} ảnh REAL từ Replay-Attack.")
    
    if missing_real_ids:
        print(f"\n[QUAN TRỌNG] Không tìm thấy video gốc (REAL) cho {len(missing_real_ids)} bản ghi trong hệ thống file hiện tại.")
        print("Các ID bị thiếu (cần tìm trong dataset gốc hoặc giải nén thêm):")
        # In ra 5 ID đầu tiên làm mẫu
        for mid in missing_real_ids[:5]:
            print(f"  - {mid}")
        print("... (và các ID khác)")
        print("\n[GỢI Ý]: Kiểm tra xem bạn đã giải nén phần 'real' của dataset Replay-Attack chưa?")
        print("Thường nó nằm ở dạng file tar/zip riêng hoặc trong folder 'real' bên cạnh 'samples'.")
        print("Nếu không tìm thấy, hệ thống sẽ tự động cân bằng dữ liệu Real từ CASIA, OULU, SiW.")
    else:
        print("[SUCCESS] Đã tìm thấy và trích xuất toàn bộ video REAL!")


def load_data_smart():
    images = []
    labels = []
    
    # Bước 1: Chuẩn bị dữ liệu Replay-Attack từ CSV
    prepare_replay_attack()
    
    print(f"[INFO] Đang quét toàn bộ dữ liệu từ: {DATASET_ROOT}...")
    
    for root, dirs, files in os.walk(DATASET_ROOT):
        # Bỏ qua các folder trung gian không chứa ảnh đã xử lý
        if 'samples' in root and 'Replay-Attack' in root:
            continue
            
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                full_path = os.path.join(root, file)
                path_lower = full_path.lower()
                
                label = None
                
                # Logic xác định nhãn
                if 'false' in path_lower or 'spoof' in path_lower or 'fake' in path_lower or 'attack' in path_lower or 'print' in path_lower or 'replay' in path_lower:
                    if 'real' not in path_lower and 'true' not in path_lower:
                        label = 0
                
                if label is None:
                    if 'true' in path_lower or 'real' in path_lower or 'live' in path_lower or 'bonafide' in path_lower:
                        label = 1
                
                # Xử lý đặc biệt cho các folder đã tách sẵn
                if 'REPLAY_FAKE' in root: label = 0
                if 'REPLAY_REAL' in root: label = 1
                
                if label is not None:
                    img = cv2.imread(full_path)
                    if img is not None:
                        # --- SỬA LỖI Ở ĐÂY: Resize ngay khi đọc để đảm bảo đồng nhất ---
                        # Resize về đúng kích thước IMG_SIZE (80x80) để tránh lỗi NumPy
                        img_resized = cv2.resize(img, IMG_SIZE)
                        
                        images.append(img_resized)
                        labels.append(label)
    
    if len(images) == 0:
        raise ValueError("Không tìm thấy ảnh nào! Hãy đảm bảo đã chạy script này để nó tự trích xuất video từ CSV trước khi load.")

    print(f"[SUCCESS] Đã load tổng cộng {len(images)} ảnh (đã resize về {IMG_SIZE}).")
    print(f"[STATS] Real: {sum(labels)} | Fake: {len(labels) - sum(labels)}")
    
    # Bây giờ mới chuyển sang NumPy Array vì tất cả ảnh đã cùng kích thước
    return np.array(images), np.array(labels)

class RealTimeLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        self.history = []
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        if os.path.exists(log_path):
            os.remove(log_path)

    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc):
        row = {
            "Epoch": epoch + 1,
            "Train_Loss": f"{train_loss:.4f}",
            "Train_Acc": f"{train_acc:.4f}",
            "Val_Loss": f"{val_loss:.4f}",
            "Val_Acc": f"{val_acc:.4f}",
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.history.append(row)
        try:
            df = pd.DataFrame(self.history)
            df.to_excel(self.log_path, index=False, sheet_name='RealTime_Log')
            print(f"\r[LOG] Epoch {epoch+1}: Acc={train_acc:.4f} | Val={val_acc:.4f}", end='')
        except Exception as e:
            print(f"\n[WARN] Lỗi ghi Excel: {e}")

def train_model():
    print("="*70)
    print("TRAINING ROBUST LIVENESS (CASIA + REPLAY+CSV + OULU + SiW)")
    print("="*70)
    
    try:
        X, y = load_data_smart()
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val)

    print(f"[DATA] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    train_dataset = LivenessDataset(X_train, y_train, transform=train_transform)
    val_dataset = LivenessDataset(X_val, y_val, transform=val_transform)
    test_dataset = LivenessDataset(X_test, y_test, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    logger = RealTimeLogger(LOG_EXCEL_PATH)
    best_val_acc = 0.0
    
    print("\n[BEGIN TRAINING]...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            preds = (outputs > 0.5).float()
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total
        
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                preds = (outputs > 0.5).float()
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        scheduler.step(epoch_val_acc)
        logger.log_epoch(epoch, epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc)
        
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    # Test
    print("\n\n[BEGIN TESTING]...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze()
            preds = (outputs > 0.5).int().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)
    
    print("\n" + "="*50)
    print("KẾT QUẢ CUỐI CÙNG:")
    print("="*50)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(classification_report(all_targets, all_preds, target_names=['Fake', 'Real']))

if __name__ == "__main__":
    train_model()