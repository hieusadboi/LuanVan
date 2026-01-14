import os
import shutil
import random
import yaml

# --- 1. CẤU HÌNH ĐƯỜNG DẪN CỦA BẠN ---
# Đường dẫn gốc chứa dữ liệu thô
RAW_IMAGES_DIR = r'D:\LuanVan\Dataset\images'
RAW_LABELS_DIR = r'D:\LuanVan\Dataset\label'  # Folder hiện tại của bạn là 'label'

# Đường dẫn đích sau khi chia (YOLO sẽ đọc từ đây)
DEST_DIR = r'D:\LuanVan\Dataset_YOLO_Standard'

def setup_dirs():
    # Tạo cấu trúc chuẩn: train/images, train/labels...
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(DEST_DIR, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(DEST_DIR, split, 'labels'), exist_ok=True)

def split_dataset():
    # Lấy danh sách ảnh
    if not os.path.exists(RAW_IMAGES_DIR):
        print(f"❌ Lỗi: Không tìm thấy thư mục {RAW_IMAGES_DIR}")
        return

    all_images = [f for f in os.listdir(RAW_IMAGES_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    # Xáo trộn và chia tỉ lệ 80 - 10 - 10
    random.seed(42)
    random.shuffle(all_images)
    
    total = len(all_images)
    train_count = int(total * 0.8)
    val_count = int(total * 0.1)
    
    train_files = all_images[:train_count]
    val_files = all_images[train_count:train_count+val_count]
    test_files = all_images[train_count+val_count:]
    
    print(f"Tổng ảnh: {total}")
    print(f"Phân chia: Train={len(train_files)} | Val={len(val_files)} | Test={len(test_files)}")
    
    # Hàm copy file
    def copy_files(files, split_name):
        print(f"⏳ Đang xử lý tập {split_name}...")
        for img_name in files:
            # 1. Copy Ảnh
            src_img = os.path.join(RAW_IMAGES_DIR, img_name)
            dst_img = os.path.join(DEST_DIR, split_name, 'images', img_name)
            shutil.copy(src_img, dst_img)
            
            # 2. Copy Nhãn (đổi từ folder 'label' sang 'labels')
            lbl_name = os.path.splitext(img_name)[0] + '.txt'
            src_lbl = os.path.join(RAW_LABELS_DIR, lbl_name)
            dst_lbl = os.path.join(DEST_DIR, split_name, 'labels', lbl_name)
            
            if os.path.exists(src_lbl):
                shutil.copy(src_lbl, dst_lbl)
            else:
                print(f"⚠️ Cảnh báo: Ảnh {img_name} không có file nhãn!")

    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')

    # --- TẠO FILE DATA.YAML TỰ ĐỘNG ---
    # Dùng đường dẫn tuyệt đối (Absolute Path) để tránh lỗi trên Windows
    data_yaml_content = {
        'path': DEST_DIR,
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 1,
        'names': ['Bien_hieu'] 
    }
    
    yaml_path = os.path.join(r'D:\LuanVan', 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml_content, f, default_flow_style=False)
    
    print(f"\n✅ Xong! Đã tạo dữ liệu chuẩn tại: {DEST_DIR}")
    print(f"✅ File cấu hình đã tạo tại: {yaml_path}")

if __name__ == "__main__":
    setup_dirs()
    split_dataset()