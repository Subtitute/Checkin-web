# File: src/notifier.py
import requests
from src.config import Config

class TelegramNotifier:
    def __init__(self):
        self.token = Config.TELEGRAM_BOT_TOKEN
        self.chat_id = Config.TELEGRAM_CHAT_ID
        # Chỉ bật nếu người dùng đã điền token và chat_id
        self.enabled = bool(self.token and self.chat_id)
        
        if self.enabled:
            print("[INFO] Module thông báo Telegram đã được kích hoạt.")
        else:
            print("[WARN] Chưa cấu hình Telegram. Tính năng thông báo sẽ bị tắt.")

    def send_message(self, message):
        if not self.enabled:
            return
        
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "HTML"
        }
        
        try:
            # Timeout ngắn để không làm giật ứng dụng chính nếu mạng chậm
            response = requests.post(url, json=payload, timeout=3)
            if response.status_code == 200:
                print("[NOTIFY] Đã gửi thông báo thành công.")
            else:
                print(f"[ERROR] Lỗi gửi Telegram: {response.text}")
        except Exception as e:
            print(f"[ERROR] Ngoại lệ khi gửi Telegram: {e}")

# Khởi tạo đối tượng toàn cục
notifier = TelegramNotifier()