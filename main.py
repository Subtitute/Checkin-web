import os
import cv2
import pandas as pd
from datetime import datetime
from src.config import Config
from src.preprocess_faces import preprocess_all_images
from src.database_builder import build_face_database
from core_checkin import run_checkin_logic
from src.export_excel import export_attendance_to_excel
import unicodedata
import re

# --- HÀM TIỆN ÍCH: Bỏ dấu tiếng Việt ---
def bo_dau_tieng_viet(text):
    """
    Chuyển đổi chuỗi có dấu thành không dấu để làm tên file an toàn.
    Ví dụ: 'Nguyễn Văn A' -> 'Nguyen_Van_A'
    """
    if not text:
        return ""
    # Chuẩn hóa unicode và loại bỏ dấu
    text = unicodedata.normalize('NFC', text)
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('utf-8')
    # Loại bỏ ký tự đặc biệt, chỉ giữ chữ, số và khoảng trắng
    text = re.sub(r'[^\w\s-]', '', text)
    return text.strip().replace(' ', '_')

def chup_anh_dang_ky():
    """
    Chức năng đăng ký người mới:
    1. Nhập thông tin.
    2. Chụp ảnh từ camera.
    3. Lưu ảnh (tên không dấu) vào data/raw.
    4. Cập nhật data.csv (tên có dấu).
    5. Hỏi chạy Preprocess và Build Database ngay lập tức.
    """
    print("\n--- CHẾ ĐỘ ĐĂNG KÝ NGƯỜI MỚI ---")
    pid = input("Nhập Mã Sinh Viên/Nhân Viên (VD: A01): ").strip()
    name = input("Nhập Họ và Tên: ").strip()
    
    if not pid or not name:
        print("Lỗi: Thiếu thông tin mã hoặc tên!")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không mở được camera! Vui lòng kiểm tra kết nối.")
        return

    print("Đặt khuôn mặt vào khung hình... Nhấn SPACE để chụp, 'q' để thoát.")
    
    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        
        h, w, _ = frame.shape
        # Vẽ khung hướng dẫn
        cv2.rectangle(frame, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 0), 2)
        cv2.putText(frame, "Nhan SPACE de chup", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Dang Ky Khuon Mat", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):
            # 1. Tạo tên file KHÔNG DẤU để tránh lỗi OpenCV đọc đường dẫn
            name_no_dau = bo_dau_tieng_viet(name)
            filename = f"{pid}_{name_no_dau}.jpg"
            
            filepath = os.path.join(Config.raw_image_dir, filename)
            os.makedirs(Config.raw_image_dir, exist_ok=True)
            
            # Lưu ảnh
            cv2.imwrite(filepath, frame)
            print(f"Đã chụp ảnh thành công: {filepath}")
            
            # 2. Cập nhật data.csv (Lưu tên CÓ DẤU để hiển thị đẹp)
            try:
                if not os.path.exists(Config.csv_info_file) or os.path.getsize(Config.csv_info_file) == 0:
                    df_new = pd.DataFrame(columns=["ID_Name", "Name"])
                else:
                    df_new = pd.read_csv(Config.csv_info_file)
                    # Đảm bảo cột cần thiết tồn tại
                    if "ID_Name" not in df_new.columns or "Name" not in df_new.columns:
                        df_new = pd.DataFrame(columns=["ID_Name", "Name"])
            except Exception as e:
                print(f"[WARN] Lỗi đọc CSV, tạo mới: {e}")
                df_new = pd.DataFrame(columns=["ID_Name", "Name"])
            
            # Kiểm tra trùng ID
            if pid in df_new["ID_Name"].values:
                print(f"Cảnh báo: ID {pid} đã tồn tại! Cập nhật lại tên...")
                df_new.loc[df_new["ID_Name"] == pid, "Name"] = name
            else:
                new_row = pd.DataFrame([{"ID_Name": pid, "Name": name}])
                df_new = pd.concat([df_new, new_row], ignore_index=True)
            
            # Lưu lại file CSV
            df_new.to_csv(Config.csv_info_file, index=False, encoding="utf-8-sig")
            print("Đã cập nhật thông tin vào data.csv")
            
            # 3. Hỏi xây dựng Database ngay
            choice = input("Xây dựng Database ngay bây giờ? (y/n): ")
            if choice.lower() == 'y':
                print("\n[BƯỚC 1/2] Đang tiền xử lý ảnh (Cắt mặt)...")
                preprocess_all_images()
                
                print("\n[BƯỚC 2/2] Đang xây dựng Database (Trích xuất đặc trưng)...")
                build_face_database()
                
                print("\n==> HOÀN TẤT! Bạn có thể bắt đầu điểm danh ngay.")
            else:
                print("\n[LƯU Ý] Bạn chưa build database. Hãy chạy 'python build_face_db.py' trước khi điểm danh.")
            
            cap.release()
            cv2.destroyAllWindows()
            return

        if key == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

def show_menu():
    """
    Hiển thị menu chính và điều hướng chức năng.
    """
    while True:
        now = datetime.now().strftime("%H:%M")
        print("\n" + "="*50)
        print("=== HỆ THỐNG CHẤM CÔNG AI (FACE RECOGNITION) ===")
        print("="*50)
        print(f"Giờ hệ thống: {now}")
        print(f"Khung giờ làm: {Config.WORK_START_TIME} - {Config.WORK_END_TIME}")
        print("-" * 50)
        print("1. Đăng ký người mới (Chụp ảnh + Lưu thông tin)")
        print("2. Chạy chế độ CHECK-IN (Vào ca)")
        print("3. Chạy chế độ CHECK-OUT (Tan ca)")
        print("4. Chạy tự động (AUTO - Tự đổi theo giờ)")
        print("5. XUẤT BÁO CÁO EXCEL (Mới)")
        print("6. Thoát chương trình")
        print("-" * 50)
        
        choice = input("Chọn chức năng (1-6): ")
        
        if choice == '1':
            chup_anh_dang_ky()
        elif choice == '2':
            print("\n>>> Khởi động chế độ CHECK-IN...")
            run_checkin_logic(mode="CHECK_IN")
        elif choice == '3':
            print("\n>>> Khởi động chế độ CHECK-OUT...")
            run_checkin_logic(mode="CHECK_OUT")
        elif choice == '4':
            print("\n>>> Khởi động chế độ TỰ ĐỘNG (AUTO)...")
            run_checkin_logic(mode="AUTO")
        elif choice == '5':
            print("\n>>> Đang xuất báo cáo Excel...")
            export_attendance_to_excel()
            input("\n[Nhấn Enter để quay lại menu...]")
        elif choice == '6':
            print("Đang thoát chương trình. Cảm ơn bạn đã sử dụng!")
            break
        else:
            print("Lựa chọn không hợp lệ! Vui lòng chọn lại (1-6).")

if __name__ == "__main__":
    # Kiểm tra nhanh các file cấu hình cần thiết
    if not os.path.exists(Config.csv_info_file):
        print(f"[INFO] Chưa tìm thấy file {Config.csv_info_file}. Hệ thống sẽ tự tạo khi đăng ký người mới.")
    
    show_menu()