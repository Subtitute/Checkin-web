# File: process_replay_labels.py
import os
import shutil
import pandas as pd
import cv2
from tqdm import tqdm

# Cấu hình
CSV_PATH = "datasets/Replay-Attack/replay_30.csv" # Đường dẫn đến file csv bạn mới up
SAMPLES_DIR = "datasets/Replay-Attack/samples"
OUTPUT_REAL = "datasets/REPLAY_REAL"
OUTPUT_FAKE = "datasets/REPLAY_FAKE"

# Tạo folder đích
os.makedirs(OUTPUT_REAL, exist_ok=True)
os.makedirs(OUTPUT_FAKE, exist_ok=True)

def extract_folder_id_from_link(link):
    """Tách ID folder từ đường dẫn link trong CSV.
    Ví dụ: '0001ffba3c--6287e2f608ea00759b5c156e/replay_video.mp4' 
    -> Trả về: '0001ffba3c--6287e2f608ea00759b5c156e'
    """
    if not link or pd.isna(link):
        return None
    return link.split('/')[0]

def process_replay_dataset():
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] Không tìm thấy file CSV tại: {CSV_PATH}")
        return

    print("[INFO] Đang đọc file label...")
    df = pd.read_csv(CSV_PATH)
    
    # Tập hợp các ID là FAKE (nằm trong cột link)
    fake_ids = set()
    for link in df['link']:
        fid = extract_folder_id_from_link(link)
        if fid:
            fake_ids.add(fid)
            
    print(f"[INFO] Đã xác định {len(fake_ids)} video FAKE từ file CSV.")
    
    # Duyệt qua tất cả folder trong samples
    if not os.path.exists(SAMPLES_DIR):
        print(f"[ERROR] Không tìm thấy thư mục samples: {SAMPLES_DIR}")
        return

    count_real = 0
    count_fake = 0
    count_unknown = 0
    
    print("\n[BEGIN PROCESSING] Đang phân loại và cắt ảnh...")
    
    for folder_name in tqdm(os.listdir(SAMPLES_DIR)):
        src_folder = os.path.join(SAMPLES_DIR, folder_name)
        if not os.path.isdir(src_folder):
            continue
            
        # Xác định nhãn
        if folder_name in fake_ids:
            label = "FAKE"
            dst_dir = OUTPUT_FAKE
            count_fake += 1
        else:
            # Giả sử các folder còn lại trong samples mà không có trong list fake 
            # thì khả năng cao là REAL (hoặc không nằm trong danh sách 30 cái này)
            # Để an toàn cho đồ án, ta chỉ lấy những cái chắc chắn.
            # Nếu folder_name trùng với live_video_id thì là REAL.
            # Tuy nhiên, thường thì structure của Replay-Attack là: 
            # Folder tên là ID của video gốc (Real) hoặc ID của video attack.
            # Dựa vào CSV này, cột 'link' chứa đường dẫn tới video attack.
            # Vậy folder trùng với phần đầu của 'link' chính là FOLDER CHỨA VIDEO GIẢ.
            
            # Còn video thật (Real) ở đâu? 
            # Thường dataset Replay-Attack gốc có cấu trúc riêng. 
            # Nếu trong folder samples này chỉ chứa các video attack được quay lại,
            # thì có thể video gốc (Real) không nằm ở đây hoặc nằm ở folder khác.
            
            # GIẢ ĐỊNH AN TOÀN NHẤT CHO TRƯỜNG HỢP CỦA BẠN:
            # Nếu folder_name CÓ TRONG fake_ids -> Là Fake.
            # Nếu folder_name KHÔNG CÓ TRONG fake_ids -> Ta tạm coi là Unknown hoặc Real (tùy dữ liệu thực tế).
            # ĐỂ ĐƠN GIẢN CHO ĐỒ ÁN: Ta chỉ lấy chắc chắn Fake trước.
            # Còn Real: Bạn cần kiểm tra xem có folder nào trùng với 'live_video_id' không.
            
            # Kiểm tra xem có trùng live_video_id không
            is_real = folder_name in df['live_video_id'].values
            
            if is_real:
                label = "REAL"
                dst_dir = OUTPUT_REAL
                count_real += 1
            else:
                # Nếu không phải Fake cũng không phải Real rõ ràng trong list này
                # Có thể đây là video gốc chưa được cặp đôi trong file csv 30 dòng này
                # Hoặc dữ liệu của bạn chỉ toàn là Attack.
                # Tạm thời bỏ qua hoặc gán vào Real nếu bạn chắc chắn samples chứa cả 2 loại.
                # Ở đây tôi sẽ gán vào REAL nếu nó không nằm trong list Fake (giả thuyết samples chứa hỗn hợp)
                # NHƯNG CẨN THẬN HƠN: Chỉ lấy những gì xác định được.
                continue 

        # Nếu xác định được nhãn, tiến hành cắt video thành ảnh
        # Tìm file video trong folder
        video_files = [f for f in os.listdir(src_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
        if not video_files:
            continue
            
        video_path = os.path.join(src_folder, video_files[0])
        
        # Cắt lấy 5 frame giữa video
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            continue
            
        indices = [int(total_frames * i / 6) for i in range(1, 6)]
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                out_name = f"{folder_name}_frame_{idx}_{label}.jpg"
                cv2.imwrite(os.path.join(dst_dir, out_name), frame)
        
        cap.release()

    print("\n=== KẾT QUẢ ===")
    print(f"Đã xử lý ảnh REAL: {count_real} video -> {count_real * 5} ảnh (ước tính)")
    print(f"Đã xử lý ảnh FAKE: {count_fake} video -> {count_fake * 5} ảnh (ước tính)")
    print(f"Dữ liệu đã được lưu vào: {OUTPUT_REAL} và {OUTPUT_FAKE}")

if __name__ == "__main__":
    process_replay_dataset()