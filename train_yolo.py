from ultralytics import YOLO
import torch
import os
import shutil
from datetime import datetime

def main():
    # --- 1. KIỂM TRA THIẾT BỊ ---
    # Tự động chọn GPU nếu có, nếu không thì dùng CPU
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Thiết bị Training: {'GPU (Nhanh)' if device == 0 else 'CPU (Chậm)'}")

    # --- 2. CẤU HÌNH MODEL ---
    # Sử dụng bản Nano (yolov8n) nhẹ nhất để train nhanh
    model_name = 'yolov8s.pt' 
    model = YOLO(model_name) 

    # --- 3. BẮT ĐẦU TRAIN (CÓ DỪNG SỚM) ---
    print("🚀 Đang bắt đầu quá trình train...")
    
    # Huấn luyện model
    model.train(
        data='data.yaml',   # File cấu hình dữ liệu của bạn
        epochs=50,          # Số vòng lặp tối đa
        imgsz=640,          # Kích thước ảnh (giảm xuống 320 cho nhẹ máy)
        batch=4,            # Số ảnh xử lý cùng lúc (giảm nếu tràn RAM)
        workers=1,          # Quan trọng cho Windows để tránh lỗi
        device=device,      # Thiết bị đã chọn ở trên
        name='DoAn_Result', # Tên thư mục kết quả trong runs/detect
        
        # === TÍNH NĂNG 1: DỪNG SỚM (EARLY STOPPING) ===
        patience=5,         # Nếu 5 epoch liên tiếp không tốt hơn -> Dừng ngay
        # ==============================================
        
        val=True,           # Có kiểm thử sau mỗi epoch
        exist_ok=True       # Cho phép ghi đè thư mục nếu cần (tùy chọn)
    )
    print("🎉 Train hoàn tất (hoặc đã dừng sớm do patience)!")

    # --- 4. XỬ LÝ KẾT QUẢ & LƯU MODEL TỐT NHẤT ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M") 
    
    # Tìm đường dẫn đến folder vừa train xong
    # Lưu ý: Ultralytics tạo folder dạng runs/detect/DoAn_Result (hoặc DoAn_Result2...)
    base_run_dir = os.path.join(os.getcwd(), 'runs', 'detect')
    
    if not os.path.exists(base_run_dir):
        print("⚠️ Không tìm thấy thư mục runs/detect.")
        return

    # Lấy folder mới nhất vừa được tạo ra
    all_subdirs = [os.path.join(base_run_dir, d) for d in os.listdir(base_run_dir) if os.path.isdir(os.path.join(base_run_dir, d))]
    if not all_subdirs:
        print("⚠️ Không tìm thấy thư mục kết quả nào.")
        return
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    print(f"📂 Thư mục kết quả gốc: {latest_subdir}")

    # === TÍNH NĂNG 2: LƯU MODEL TỐT NHẤT (BEST.PT) ===
    src_best_path = os.path.join(latest_subdir, 'weights', 'best.pt')
    dst_best_name = f"BEST_MODEL_{timestamp}.pt" # Đặt tên file dễ nhớ
    dst_best_path = os.path.join(os.getcwd(), dst_best_name)

    if os.path.exists(src_best_path):
        shutil.copy(src_best_path, dst_best_path)
        print(f"\n✅ ĐÃ LƯU MODEL TỐT NHẤT TẠI: {dst_best_path}")
    else:
        print(f"⚠️ Không tìm thấy file best.pt (Có thể quá trình train bị lỗi giữa chừng)")

    # === TÍNH NĂNG 3: TRÍCH XUẤT BÁO CÁO HÌNH ẢNH ===
    report_folder = os.path.join(os.getcwd(), f"BaoCao_Anh_{timestamp}")
    os.makedirs(report_folder, exist_ok=True)
    
    # Danh sách các ảnh biểu đồ quan trọng cần lấy
    files_to_copy = [
        'confusion_matrix.png',      # Ma trận nhầm lẫn
        'results.png',               # Biểu đồ Loss/Accuracy
        'PR_curve.png',              # Biểu đồ Precision-Recall
        'val_batch0_labels.jpg',     # Ảnh nhãn thực tế
        'val_batch0_pred.jpg'        # Ảnh model dự đoán (để so sánh)
    ]

    print(f"\n📊 Đang trích xuất báo cáo vào thư mục: {report_folder}")
    count = 0
    for file_name in files_to_copy:
        src = os.path.join(latest_subdir, file_name)
        dst = os.path.join(report_folder, file_name)
        if os.path.exists(src):
            shutil.copy(src, dst)
            count += 1
            
    print(f"✅ Đã copy {count} ảnh báo cáo.")
    print(f"👉 Hãy mở folder '{report_folder}' để xem biểu đồ.")

if __name__ == '__main__':
    model_check = 'yolov8n.pt'
    if os.path.exists(model_check) and os.path.getsize(model_check) < 1000:
        print("⚠️ File model gốc bị lỗi, đang xóa để tải lại...")
        os.remove(model_check)
        
    main()