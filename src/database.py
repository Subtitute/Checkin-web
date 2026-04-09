# File: src/database.py
import numpy as np
import pandas as pd
import os
from src.config import Config

def load_database():
    """
    Nạp database embeddings và thông tin nhãn từ file.
    Returns:
        db_embeddings (np.array): Ma trận vector đặc trưng.
        db_labels_df (pd.DataFrame): DataFrame chứa thông tin ID, Name.
    """
    if not os.path.exists(Config.embeddings_file):
        raise FileNotFoundError(f"Không tìm thấy file embeddings: {Config.embeddings_file}")
    
    if not os.path.exists(Config.csv_info_file):
        raise FileNotFoundError(f"Không tìm thấy file thông tin: {Config.csv_info_file}")

    embeddings = np.load(Config.embeddings_file)
    labels_df = pd.read_csv(Config.csv_info_file)
    
    # Xử lý lỗi nếu file CSV bị rỗng hoặc hỏng
    if labels_df.empty:
        print("[WARN] File CSV rỗng. Khởi tạo lại cấu trúc.")
        labels_df = pd.DataFrame(columns=['ID_Name', 'Name'])
    
    return embeddings, labels_df

def calculate_cosine_similarity(vec, mat):
    """
    Tính toán cosine similarity giữa 1 vector và một ma trận.
    """
    vec_norm = vec / np.linalg.norm(vec)
    mat_norm = mat / np.linalg.norm(mat, axis=1, keepdims=True)
    return mat_norm @ vec_norm

def find_best_match(embedding, db_embeddings, db_labels_df, threshold=None):
    """
    Tìm khuôn mặt khớp nhất trong database.
    
    Returns:
        tuple: (match_dict, best_sim)
            - match_dict: Dictionary chứa thông tin người khớp (hoặc None).
            - best_sim: Điểm similarity cao nhất.
    """
    if threshold is None:
        threshold = Config.threshold
        
    if len(db_embeddings) == 0:
        return None, 0.0

    sims = calculate_cosine_similarity(embedding, db_embeddings)
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])
    
    if best_sim >= threshold:
        best_match_row = db_labels_df.iloc[best_idx]
        
        # Chuyển đổi sang Dictionary để tránh lỗi Pandas Series
        match_dict = best_match_row.to_dict()
        
        # Đảm bảo kiểu dữ liệu string
        if 'ID_Name' in match_dict:
            match_dict['ID_Name'] = str(match_dict['ID_Name'])
        if 'Name' in match_dict:
            match_dict['Name'] = str(match_dict['Name'])
            
        return match_dict, best_sim
    else:
        return None, best_sim

def add_new_face_to_db(new_id, new_name, new_embedding):
    """
    Thêm người dùng mới vào database hiện có.
    
    Args:
        new_id (str): Mã nhân viên/sinh viên.
        new_name (str): Họ tên.
        new_embedding (np.array): Vector đặc trưng khuôn mặt (1D).
    """
    # 1. Xử lý Embeddings
    # Đảm bảo new_embedding có shape (1, D)
    if new_embedding.ndim == 1:
        new_embedding = new_embedding.reshape(1, -1)
    
    if os.path.exists(Config.embeddings_file):
        old_embeddings = np.load(Config.embeddings_file)
        # Nối vector mới vào cuối
        updated_embeddings = np.vstack([old_embeddings, new_embedding])
    else:
        updated_embeddings = new_embedding

    # Lưu lại file .npy
    np.save(Config.embeddings_file, updated_embeddings)

    # 2. Xử lý CSV Information
    new_row = {'ID_Name': new_id, 'Name': new_name}
    
    if os.path.exists(Config.csv_info_file):
        df = pd.read_csv(Config.csv_info_file)
        # Kiểm tra trùng ID
        if new_id in df['ID_Name'].values:
            raise ValueError(f"Mã {new_id} đã tồn tại trong hệ thống!")
        
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])
    
    # Lưu lại file .csv
    df.to_csv(Config.csv_info_file, index=False)
    
    print(f"[DATABASE] Đã thêm thành công: {new_name} ({new_id}). Tổng số người: {len(df)}")