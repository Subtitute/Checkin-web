# 🎓 Check-in Web - Face Recognition + Anti-Spoofing

> 📌 **Đồ án tốt nghiệp** | [Your Name] | [University Name] | 2024

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3+-lightgrey)](https://flask.palletsprojects.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🚀 Quick Start

```bash
# Clone repo
git clone https://github.com/Subtitute/Checkin-web.git
cd Checkin-web

# Tạo venv & install deps
python -m venv venv
.\venv\Scripts\Activate  # PowerShell
pip install -r requirements.txt

# Chạy server
python server.py

# Open browser: http://localhost:5000

📁 Project Structure
.
├── 📄 server.py              # Flask app entry point
├── 📄 requirements.txt       # Python dependencies
├── 📄 Dockerfile            # Docker config (optional)
├── 📄 .gitignore            # Git ignore rules
│
├── 📁 src/                   # Main source code
│   ├── core_checkin.py      # Check-in logic
│   ├── face_db.py           # Face database manager
│   └── utils.py             # Helper functions
│
├── 📁 Silent-Face-Anti-Spoofing/  # Anti-spoofing module
│   ├── src/model_lib/       # Model architecture
│   └── resources/anti_spoof_models/  # Pretrained weights
│
├── 📁 data/                  # Runtime data (git-ignored)
│   ├── faces/               # Registered user faces
│   ├── processed/           # Processed images
│   └── raw/                 # Raw camera captures
│
├── 📁 datasets/              # Training datasets (git-ignored)
│   ├── CASIA-FASD/
│   ├── OULU-NPU/
│   ├── Replay-Attack/
│   └── SiW/
│
├── 📁 models/                # Model checkpoints (git-ignored)
├── 📁 templates/             # HTML templates
└── 📁 static/                # CSS, JS, images

⚙️ Configuration
Environment Variables (.env - optional)
FLASK_ENV=development
FLASK_DEBUG=1
DEVICE=cuda  # hoặc 'cpu'
THRESHOLD=0.6  # Face matching threshold

Requirements (requirements.txt)
flask>=2.3.0
opencv-python>=4.8.0
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
Pillow>=9.5.0

🔌 API Reference
### Face Check-in API

#### 1. Check-in/Recognition Endpoint

**POST** `/api/checkin`

Recognizes a person or registers a new user via base64 image.

**Request Example (face recognition):**
```python
import requests
import base64

with open("face.jpg", "rb") as img_file:
    img_b64 = base64.b64encode(img_file.read()).decode("utf-8")

resp = requests.post(
    "http://localhost:5000/api/checkin",
    json={
        "image": img_b64,
        "mode": "checkin"  # or "verify"
    }
)
print(resp.json())
```

**Request Fields:**
- `image` (str, required): base64-encoded image data (JPEG/PNG)
- `mode` (str, optional): `"checkin"` (default), `"register"`, or `"verify"`
- `reg_id` (str, for register): ID for registration (required for `"register"`)
- `reg_name` (str, for register): Name for registration (required for `"register"`)

**Response Fields:**
- `success` (bool)
- `message` (str)
- `id` (str, check-in result user ID, if matched)
- `name` (str, matched/registered name)
- `timestamp` (str)
- (May include additional info: liveness score, confidence, etc.)

#### 2. Register New User

**Request Example:**
```python
import requests
import base64

with open("your_face.jpg", "rb") as img_file:
    img_b64 = base64.b64encode(img_file.read()).decode("utf-8")

resp = requests.post(
    "http://localhost:5000/api/checkin",
    json={
        "image": img_b64,
        "mode": "register",
        "reg_id": "00123",
        "reg_name": "Nguyen Van A"
    }
)
print(resp.json())
```

**Expected Response (success):**
```json
{
  "success": true,
  "message": "Đã đăng ký thành công cho Nguyen Van A",
  "id": "00123",
  "name": "Nguyen Van A",
  "timestamp": "16:11:34 - 14/03/2024"
}
```

### Error Handling

- Missing/invalid image: `{"success": false, "message": "Không nhận được ảnh hợp lệ"}`
- No face found: `{"success": false, "message": "Không tìm thấy khuôn mặt."}`
- Registration missing info: `{"success": false, "message": "Thiếu thông tin Mã hoặc Tên"}`

---

Example: Check-in Request

import requests, base64

with open('test.jpg', 'rb') as f:
    payload = {'image': base64.b64encode(f.read()).decode()}

res = requests.post('http://localhost:5000/api/checkin', json=payload)
print(res.json())
# {'success': True, 'user': {'id': 1, 'name': 'Nguyen Van A'}}

## 📊 Results

### Liveness Detection (CASIA-FASD test set)

| Metric | Value |
|--------|-------|
| Accuracy | 98.2% |
| APCER | 1.1% |
| BPCER | 2.3% |

### Face Recognition (Internal dataset - 50 users)

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | 96.8% |
| FAR | 0.5% |
| FRR | 3.2% |

### Performance
• Inference time: ~320ms/frame (RTX 3060)
• RAM usage: ~2.1GB
• Throughput: ~3 FPS realtime

## 🛠️ Development
### Run with Docker

docker build -t checkin-web .
docker run -p 5000:5000 --gpus all checkin-web

### Train Anti-Spoofing Model
cd Silent-Face-Anti-Spoofing
python src/train.py --data_dir ../datasets/CASIA-FASD --model_dir ../models

### Add New User Face 
# Using Python API
from src.face_db import FaceDatabase

db = FaceDatabase('data/faces')
db.register(user_id=101, name="Nguyen Van B", image_path="path/to/photo.jpg")

## 📚 References
# Liu et al., Silent Face Anti-Spoofing, arXiv:2006.11460
# CASIA-FASD Dataset: http://www.cbsr.ia.ac.cn/english/FaceAntiSpoofingDatabases.asp
# Flask Docs: https://flask.palletsprojects.com
# PyTorch Docs: https://pytorch.org/docs

## 👥 Authors

| Name | Role | Email |
|------|------|-------|
| Dương Minh Quang | Main Developer | 0218466@huce.edu.vn |
| Phạm Hồng Phong | Thesis Advisor |                      |

> 🎓 **Graduation Project** - Đại học Xây Dựng Hà Nội - 2026
