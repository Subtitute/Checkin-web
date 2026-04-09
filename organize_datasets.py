# File: organize_datasets.py
import os
import shutil
import cv2
from tqdm import tqdm

# Cấu hình đường dẫn gốc
ROOT_DATASETS = "datasets"
OUTPUT_DIR = "datasets_organized" # Folder mới chứa dữ liệu đã sắp xếp

# Tạo cấu trúc folder đích
sub_folders = ["CASIA_REAL", "CASIA_FAKE", "REPLAY_REAL", "REPLAY_FAKE"]
for folder in sub_folders:
    os.makedirs(os.path.join(OUTPUT_DIR, folder), exist_ok=True)

def process_casia():
    """Xử lý CASIA-FASD (Dạng ảnh có đuôi _real/_fake)"""
    casia_path = os.path.join(ROOT_DATASETS, "CASIA-FASD")
    if not os.path.exists(casia_path):
        print(f"[SKIP] Không tìm thấy folder {casia_path}")
        return

    print("[INFO] Đang xử lý CASIA-FASD...")
    
    # Duyệt qua train_img và test_img
    for data_type in ["train_img", "test_img"]:
        type_path = os.path.join(casia_path, data_type)
        if not os.path.exists(type_path): continue
        
        # Duyệt qua color và depth (ta chỉ lấy color để đơn giản hóa, hoặc lấy cả 2)
        for mod in ["color", "depth"]:
            mod_path = os.path.join(type_path, mod)
            if not os.path.exists(mod_path): continue
            
            count_real = 0
            count_fake = 0
            
            for fname in os.listdir(mod_path):
                if not fname.endswith('.jpg'): continue
                
                src_path = os.path.join(mod_path, fname)
                
                # Phân loại dựa vào đuôi tên file như hình bạn gửi
                if "_real" in fname.lower():
                    dst_folder = os.path.join(OUTPUT_DIR, "CASIA_REAL")
                    count_real += 1
                elif "_fake" in fname.lower():
                    dst_folder = os.path.join(OUTPUT_DIR, "CASIA_FAKE")
                    count_fake += 1
                else:
                    continue # Bỏ qua file không rõ nhãn
                
                # Copy file sang folder mới (Đổi tên để tránh trùng lặp nếu có)
                # Tên mới: CASIA_[TênGốc]
                new_name = f"CASIA_{data_type}_{mod}_{fname}"
                dst_path = os.path.join(dst_folder, new_name)
                shutil.copy2(src_path, dst_path)
            
            print(f"   - {data_type}/{mod}: Real={count_real}, Fake={count_fake}")

def process_replay_attack():
    """Xử lý Replay-Attack (Dạng Video)"""
    replay_path = os.path.join(ROOT_DATASETS, "Replay-Attack")
    if not os.path.exists(replay_path):
        print(f"[SKIP] Không tìm thấy folder {replay_path}")
        return

    print("[INFO] Đang xử lý Replay-Attack (Cắt video thành ảnh)...")
    
    samples_path = os.path.join(replay_path, "samples")
    if not os.path.exists(samples_path): return

    # Quy tắc đặt tên của Replay-Attack:
    # Các folder bắt đầu bằng '0001' thường là Client (Real) - Cần kiểm tra kỹ file list gốc
    # Các folder bắt đầu bằng '0002' thường là Attack (Fake)
    # TẠM THỜI TA DỰA VÀO TÊN FOLDER HOẶC BẠN CẦN FILE LIST ĐỂ CHÍNH XÁC 100%
    # Giả sử ở đây ta dùng quy tắc phổ biến: 
    # Nếu tên folder chứa 'client' -> Real, chứa 'attack' -> Fake. 
    # NHƯNG trong tree bạn gửi chỉ có mã hex. 
    # => GIẢI PHÁP AN TOÀN: Ta sẽ cắt tất cả video, lưu vào 1 chỗ tạm, sau đó bạn cần file chú thích để tách.
    
    # TUY NHIÊN, để demo nhanh, tôi sẽ giả lập logic phân loại dựa trên ID (bạn cần đối chiếu với tài liệu dataset)
    # Thông thường: 
    # Real: Các video quay trực tiếp người thật.
    # Fake: Các video quay lại màn hình/in ấn.
    # Do không có file list trong tay lúc này, tôi sẽ viết code cắt video ra ảnh và lưu kèm theo ID folder.
    # SAU ĐÓ BẠN SẼ CHẠY THÊM 1 BƯỚC NHỎ ĐỂ GÁN NHÃN HOẶC TÔI SẼ CẬP NHẬT LẠI KHI BẠN CÓ QUY TẮC TÊN.
    
    # CẬP NHẬT: Dựa vào tài liệu Replay-Attack, các sample thường được phân loại trong file text đi kèm.
    # Vì không có file đó, ta sẽ cắt video ra ảnh và lưu vào folder 'REPLAY_RAW'.
    # Sau đó dùng script khác để tách Real/Fake dựa trên file list.
    
    raw_replay_dir = os.path.join(OUTPUT_DIR, "REPLAY_RAW")
    os.makedirs(raw_replay_dir, exist_ok=True)
    
    total_videos = len([d for d in os.listdir(samples_path) if os.path.isdir(os.path.join(samples_path, d))])
    
    for folder_name in tqdm(os.listdir(samples_path), total=total_videos, desc="Cắt video Replay"):
        vid_folder = os.path.join(samples_path, folder_name)
        if not os.path.isdir(vid_folder): continue
        
        # Tìm file video trong folder (thường là .avi hoặc .mov)
        video_files = [f for f in os.listdir(vid_folder) if f.endswith(('.avi', '.mov', '.mp4'))]
        if not video_files: continue
        
        video_path = os.path.join(vid_folder, video_files[0])
        
        # Cắt video: Lấy 10 frame giữa video để đại diện
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0: continue
        
        # Lấy 5 frame cách đều nhau
        indices = [int(frame_count * i / 6) for i in range(1, 6)]
        
        saved_count = 0
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Lưu ảnh với tên chứa ID folder để dễ phân loại sau
                # Ví dụ: REPLAY_0001ffba3c..._frame_1.jpg
                out_name = f"REPLAY_{folder_name}_frame_{idx}.jpg"
                cv2.imwrite(os.path.join(raw_replay_dir, out_name), frame)
                saved_count += 1
        cap.release()

    print(f"\n[HOÀN TẤT] Đã cắt video Replay-Attack thành ảnh vào folder: {raw_replay_dir}")
    print("[LƯU Ý QUAN TRỌNG]: Bạn cần file chú thích (list) của Replay-Attack để biết folder nào là Real/Fake.")
    print("Sau khi có danh sách, hãy chạy script 'finalize_replay_labels.py' (sẽ cung cấp bên dưới) để tách vào REAL/FAKE.")

if __name__ == "__main__":
    process_casia()
    process_replay_attack()
    print("\n=== KẾT THÚC QUÁ TRÌNH TỔ CHỨC DỮ LIỆU ===")