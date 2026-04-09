# File: core_checkin.py
import cv2
from datetime import datetime
from collections import defaultdict
import time
from src.config import Config
from src.face_model import init_insightface, extract_faces, extract_embedding
from src.database import load_database, find_best_match
from src.checkin import init_checkin_log, draw_result, log_checkin, get_work_status
from src.liveness_check import check_liveness  # Module chống giả mạo đã cập nhật

def get_latest_frame(cap):
    """
    Đọc và vứt bỏ các khung hình cũ trong bộ đệm, chỉ trả về khung hình mới nhất.
    Kỹ thuật này giúp:
    1. Giảm độ trễ (latency) xuống mức thấp nhất (gần như thời gian thực).
    2. Tránh tràn bộ đệm gây lỗi 'corrupted macroblock' hoặc 'no frame' sau khi chạy lâu.
    3. Tương tự cơ chế xử lý luồng video của Frigate.
    """
    if not cap.isOpened():
        return None, False
    
    latest_frame = None
    ret = False
    
    # Đọc liên tiếp tối đa 5 lần để xả sạch buffer cũ
    # Nếu camera gửi 30fps mà code xử lý chậm, các frame thừa sẽ bị bỏ qua
    for _ in range(5):
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        latest_frame = frame
    
    return latest_frame, ret

def run_checkin_logic(mode="AUTO"):
    """
    Chạy luồng chính của hệ thống điểm danh.
    mode: "CHECK_IN", "CHECK_OUT", hoặc "AUTO" (tự động dựa trên giờ làm việc)
    """
    print("[INFO] Đang khởi tạo mô hình nhận diện khuôn mặt...")
    app = init_insightface()
    
    try:
        db_embeddings, db_labels = load_database()
        print(f"[INFO] Đã tải database: {len(db_embeddings)} người.")
    except Exception as e:
        print(f"[ERROR] Không thể tải database: {e}")
        print("[HƯỚNG DẪN] Hãy chạy 'python build_face_db.py' để tạo database trước.")
        return

    # Các biến trạng thái cho từng người dùng
    pending_frames = defaultdict(int)      # Đếm số khung hình liên tiếp phát hiện mặt
    was_in_frame = defaultdict(bool)       # Trạng thái có mặt trong khung hình trước đó
    last_left_time = defaultdict(lambda: None) # Thời điểm rời khỏi khung hình
    last_action_time = defaultdict(lambda: None) # Thời điểm check-in/out gần nhất
    session_processed = defaultdict(bool)  # Flag đã xử lý trong phiên hiện tại chưa

    init_checkin_log()
    
    # --- CẤU HÌNH CAMERA TỐI ƯU CHO RTSP ---
    cam_index = Config.CAMERA_INDEX
    cam_url = Config.CAMERA_URL
    
    cap = None
    
    if cam_url:
        print(f"[INFO] Đang kết nối IP Camera: {cam_url}...")
        
        # Kỹ thuật ổn định hóa luồng RTSP:
        # 1. Ép buộc dùng TCP (rtsp_transport=tcp) để tránh mất gói tin UDP gây lỗi macroblock.
        # 2. Tăng kích thước bộ đệm (buffer_size) để chống giật mạng.
        # 3. Thêm tham số trực tiếp vào URL để tương thích mọi phiên bản OpenCV.
        separator = "&" if "?" in cam_url else "?"
        safe_url = f"{cam_url}{separator}rtsp_transport=tcp&buffer_size=1048576"
        
        cap = cv2.VideoCapture(safe_url)
        
        # Đặt thời gian chờ (timeout) để không bị treo vĩnh viễn nếu camera phản hồi chậm
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000) # 10s để kết nối
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)  # 5s để đọc mỗi frame
        
    else:
        print(f"[INFO] Đang kết nối Webcam chỉ số: {cam_index}...")
        cap = cv2.VideoCapture(cam_index)

    if not cap.isOpened():
        if cam_url:
            print("[ERROR] Không thể kết nối IP Camera. Kiểm tra lại:")
            print("   - Địa chỉ IP, Username, Password.")
            print("   - Camera và máy tính có cùng mạng LAN không?")
            print("   - Thử mở link RTSP bằng VLC Player để kiểm tra.")
        else:
            print(f"[ERROR] Không thể mở Webcam {cam_index}. Hãy thử cắm lại dây hoặc đổi CAMERA_INDEX.")
        return
    
    print("[SUCCESS] Camera đã sẵn sàng!")
    print(f"[INFO] Chế độ hoạt động: {mode}")
    print("[INFO] Nhấn 'q' để thoát\n")
    
    # Biến đếm để in log thống kê thỉnh thoảng
    frame_count = 0
    
    while True:
        # --- ĐỌC KHUNG HÌNH VỚI CƠ CHẾ XẢ BUFFER ---
        # Thay vì cap.read() thông thường, dùng hàm get_latest_frame()
        frame, ret = get_latest_frame(cap)
        
        if not ret or frame is None or frame.size == 0:
            # Chỉ in cảnh báo nếu liên tục mất tín hiệu (tránh spam log)
            if frame_count % 30 == 0:
                print("[WARN] Mất tín hiệu camera hoặc khung hình rỗng. Đang thử lại...")
            # Có thể thêm logic tự động reconnect ở đây nếu cần thiết
            time.sleep(0.1) # Nghỉ một chút trước khi thử lại
            frame_count += 1
            continue
        
        now = datetime.now()
        display_frame = frame.copy()
        frame_count += 1
        
        # Phát hiện khuôn mặt
        faces = extract_faces(app, frame)
        in_frame_now = set()
        
        # Xác định chế độ hiện hành (nếu là AUTO)
        current_mode = mode
        if mode == "AUTO":
            current_mode = get_work_status(now)
        
        for face in faces:
            # 1. Cắt ảnh khuôn mặt để kiểm tra Liveness
            x1, y1, x2, y2 = map(int, face.bbox)
            # Đảm bảo vùng cắt không bị lỗi index
            if y2 <= y1 or x2 <= x1:
                continue
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                continue

            # 2. KIỂM TRA LIVENESS (CHỐNG GIẢ MẠO)
            # Trả về (is_real: bool, score: float)
            is_real, liveness_score = check_liveness(face_crop)
            
            # In debug score thỉnh thoảng để theo dõi (mỗi 100 frame)
            # if frame_count % 100 == 0:
            #     print(f"[DEBUG] Liveness Score: {liveness_score:.4f}")

            if not is_real:
                # Nếu là ảnh giả (2D), vẽ cảnh báo và BỎ QUA không cho điểm danh
                draw_result(display_frame, face, None, 0, False, current_mode, is_spoof=True)
                continue

            # 3. Nếu là người thật, tiến hành trích xuất embedding và so khớp
            embedding = extract_embedding(face)
            match_label, similarity = find_best_match(embedding, db_embeddings, db_labels)
            
            confirmed = False
            pid = None
            
            if match_label is not None:
                pid = match_label["ID_Name"]
                in_frame_now.add(pid)
                
                # Tăng bộ đếm khung hình liên tiếp để chống nhiễu
                pending_frames[pid] = min(pending_frames.get(pid, 0) + 1, Config.confirm_frames)
                
                if pending_frames[pid] >= Config.confirm_frames:
                    confirmed = True
            else:
                confirmed = False
            
            # 4. Vẽ kết quả lên màn hình
            draw_result(display_frame, face, match_label, similarity, confirmed, current_mode, is_spoof=False)
            
            # 5. Xử lý ghi log nếu đã xác nhận đủ số khung hình
            if confirmed and pid is not None:
                allow = False
                first_time = last_action_time[pid] is None
                
                if first_time:
                    allow = True
                else:
                    # Kiểm tra thời gian chờ (cooldown) giữa 2 lần check-in/out
                    gap = (now - last_action_time[pid]).total_seconds()
                    if gap >= Config.checkin_cooldown and not session_processed[pid]:
                        allow = True
                
                if allow:
                    name = match_label["Name"]
                    log_checkin(pid, name, similarity, current_mode)
                    
                    last_action_time[pid] = now
                    session_processed[pid] = True # Đánh dấu đã xử lý
        
        # Reset trạng thái khi người dùng rời khỏi khung hình
        all_ids = set(list(was_in_frame.keys()) + list(in_frame_now))
        for pid in all_ids:
            if was_in_frame[pid] and pid not in in_frame_now:
                last_left_time[pid] = now
                pending_frames[pid] = 0
                session_processed[pid] = False # Cho phép check lại khi quay lại
            was_in_frame[pid] = (pid in in_frame_now)
            
        # Hiển thị cửa sổ
        cv2.imshow("He Thong Cham Cong AI (Robust RTSP)", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Đã đóng hệ thống.")

if __name__ == "__main__":
    # Mặc định chạy chế độ AUTO khi chạy trực tiếp file này
    run_checkin_logic(mode="AUTO")