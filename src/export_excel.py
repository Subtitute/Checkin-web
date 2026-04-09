# File: src/export_excel.py
import pandas as pd
import os
from datetime import datetime
from src.config import Config

def export_attendance_to_excel():
    """
    Đọc file check_log.csv và xuất ra file Excel báo cáo chấm công chi tiết.
    File Excel sẽ gộp các lần check-in/check-out của cùng 1 người trong ngày.
    """
    log_file = Config.checkin_log
    output_file = os.path.join(Config.processed_dir, "Bao_Cao_Cham_Cong.xlsx")
    
    if not os.path.exists(log_file):
        print(f"[WARN] Chưa có file log để xuất Excel: {log_file}")
        return

    try:
        # 1. Đọc dữ liệu từ CSV
        df = pd.read_csv(log_file)
        if df.empty:
            print("[INFO] File log rỗng, không có dữ liệu để xuất.")
            return

        # Chuyển đổi cột timestamp sang datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['Date'] = df['timestamp'].dt.date
        df['Time'] = df['timestamp'].dt.time
        
        # 2. Xử lý dữ liệu: Tách riêng Check-in và Check-out gần nhất trong ngày
        report_data = []
        
        # Nhóm theo Ngày và ID_Name
        grouped = df.groupby(['Date', 'ID_Name'])
        
        for (date, pid), group in grouped:
            name = group['Name'].iloc[0]
            # house = group['House'].iloc[0] # Nếu bạn đã bỏ cột House thì xóa dòng này hoặc để trống
            
            # Lọc các bản ghi CHECK_IN và CHECK_OUT
            ins = group[group['type'] == 'CHECK_IN']
            outs = group[group['type'] == 'CHECK_OUT']
            
            check_in_time = ins['Time'].min() if not ins.empty else None
            check_out_time = outs['Time'].max() if not outs.empty else None
            
            # Xác định trạng thái
            status = "Vắng"
            if check_in_time and check_out_time:
                status = "Đủ ca"
            elif check_in_time:
                # Kiểm tra giờ vào muộn (ví dụ sau 7:30)
                # Bạn có thể thêm logic so sánh giờ cụ thể ở đây
                status = "Đi làm (Chưa về)"
            elif check_out_time:
                status = "Chỉ điểm về"
                
            report_data.append({
                "Ngày": date,
                "Mã NV/SV": pid,
                "Họ và Tên": name,
                # "Phòng ban/Lớp": house, # Bỏ qua nếu không dùng
                "Giờ Check-In": check_in_time,
                "Giờ Check-Out": check_out_time,
                "Trạng Thái": status,
                "Tổng số lần quét": len(group)
            })
            
        df_report = pd.DataFrame(report_data)
        
        # Sắp xếp theo Ngày và Mã
        df_report = df_report.sort_values(by=["Ngày", "Mã NV/SV"])

        # 3. Xuất ra Excel
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df_report.to_excel(writer, index=False, sheet_name='Chi Tiết Chấm Công')
            
            # Tùy chỉnh độ rộng cột (Optional)
            worksheet = writer.sheets['Chi Tiết Chấm Công']
            column_widths = [12, 15, 25, 15, 15, 20, 15] # Điều chỉnh tùy ý
            for i, width in enumerate(column_widths):
                col_letter = chr(65 + i) # A, B, C...
                worksheet.column_dimensions[col_letter].width = width

        print(f"[SUCCESS] Đã xuất báo cáo Excel thành công: {output_file}")
        print(f"   - Tổng số bản ghi: {len(df_report)}")
        
    except Exception as e:
        print(f"[ERROR] Lỗi khi xuất Excel: {e}")

if __name__ == "__main__":
    export_attendance_to_excel()