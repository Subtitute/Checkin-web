import os
import cv2
import pandas as pd
from datetime import datetime
from src.config import Config
from src.export_json import export_user_json
from src.notifier import notifier  # Import module mới

checked_in_users = []

def init_checkin_log():
    os.makedirs(Config.processed_dir, exist_ok=True)
    if not os.path.exists(Config.checkin_log):
        with open(Config.checkin_log, "w", encoding="utf-8-sig") as f:
            # Đã bỏ cột House
            f.write("timestamp,ID_Name,Name,similarity,type\n")

def get_work_status(current_time):
    now_str = current_time.strftime("%H:%M")
    start_time = getattr(Config, 'WORK_START_TIME', "07:00")
    end_time = getattr(Config, 'WORK_END_TIME', "17:30")
    if start_time <= now_str < end_time:
        return "CHECK_IN"
    else:
        return "CHECK_OUT"

def _update_guest_sheet(name, ts_iso, status_type):
    if not os.path.exists(Config.guest_csv_file):
        df_new = pd.DataFrame(columns=["STT", "Name", "MSSV", "checkin_time", "checkin_count", "checkin_last", "Type"])
        df_new.to_csv(Config.guest_csv_file, index=False, encoding="utf-8-sig")
        return

    try:
        df = pd.read_csv(Config.guest_csv_file)
    except Exception:
        df = pd.DataFrame(columns=["STT", "Name", "MSSV", "checkin_time", "checkin_count", "checkin_last", "Type"])

    mask = (df["Name"] == name)
    
    if not mask.any():
        new_row = {
            "STT": len(df) + 1,
            "Name": name,
            "MSSV": "", 
            "checkin_time": ts_iso if status_type == "CHECK_IN" else "",
            "checkin_count": 1,
            "checkin_last": ts_iso,
            "Type": status_type
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        idx = df.index[mask][0]
        if status_type == "CHECK_IN":
            current_first = df.at[idx, "checkin_time"]
            if pd.isna(current_first) or str(current_first).strip() == "":
                df.at[idx, "checkin_time"] = ts_iso
        
        df.at[idx, "checkin_last"] = ts_iso
        
        try:
            current_count = df.at[idx, "checkin_count"]
            count_val = 0 if pd.isna(current_count) else int(current_count)
        except Exception:
            count_val = 0
        df.at[idx, "checkin_count"] = count_val + 1
        df.at[idx, "Type"] = status_type

    df.to_csv(Config.guest_csv_file, index=False, encoding="utf-8-sig")

def log_checkin(pid, name, similarity, status_type):
    global checked_in_users
    ts = datetime.now().isoformat(timespec="seconds")
    time_str = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
    
    # 1. Log vào file CSV
    with open(Config.checkin_log, "a", encoding="utf-8-sig") as f:
        f.write(f"{ts},{pid},{name},{similarity:.4f},{status_type}\n")
    
    # 2. Cập nhật guest sheet
    _update_guest_sheet(name, ts, status_type)
    
    # 3. Gửi thông báo Telegram (MỚI)
    action_icon = "✅" if status_type == "CHECK_IN" else "🏁"
    status_text = "ĐI LÀM" if status_type == "CHECK_IN" else "TAN CA"
    
    message = (
        f"<b>{action_icon} {status_text}</b>\n"
        f"👤 <b>Họ tên:</b> {name}\n"
        f"🆔 <b>Mã NV:</b> {pid}\n"
        f"⏰ <b>Thời gian:</b> {time_str}"
    )
    # Gọi hàm gửi (chạy bất đồng bộ trong thực tế production, nhưng ở đây gọi trực tiếp cho đơn giản)
    notifier.send_message(message)
    
    # 4. In log console
    print(f"[{status_type}] {time_str} - {pid} - {name} | Sim: {similarity:.4f}")
    
    # 5. Export JSON
    already_exists = any(u["userId"] == pid for u in checked_in_users)
    if already_exists:
        return
        
    checked_in_users.append({
        "userId": pid,
        "name": name,
        "type": status_type,
        "time": time_str
    })
    export_user_json(checked_in_users)

def draw_result(display_frame, face, match_label, similarity, confirmed, status_type, is_spoof=False):
    x1, y1, x2, y2 = face.bbox.astype(int)
    
    COLOR_UNKNOWN = (0, 0, 255)       # Đỏ
    COLOR_CONFIRMING = (0, 255, 255)  # Vàng
    COLOR_SUCCESS_IN = (0, 255, 0)    # Xanh lá
    COLOR_SUCCESS_OUT = (255, 0, 255) # Tím
    COLOR_SPOOF = (0, 0, 255)         # Đỏ đậm

    if is_spoof:
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), COLOR_SPOOF, 3)
        cv2.putText(display_frame, "SPOLF DETECTED!", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_SPOOF, 2)
        return

    if match_label is not None:
        pid = match_label["ID_Name"]
        name = match_label["Name"]
        
        if not confirmed:
            color = COLOR_CONFIRMING
            status_text = "Dang xac nhan..."
        else:
            if status_type == "CHECK_IN":
                color = COLOR_SUCCESS_IN
                status_text = f"DA CHECK-IN: {datetime.now().strftime('%H:%M:%S')}"
            else:
                color = COLOR_SUCCESS_OUT
                status_text = f"DA CHECK-OUT: {datetime.now().strftime('%H:%M:%S')}"

        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display_frame, f"{name} ({pid})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(display_frame, status_text, (x1, y2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    else:
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), COLOR_UNKNOWN, 2)
        cv2.putText(display_frame, "Unknown", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_UNKNOWN, 2)