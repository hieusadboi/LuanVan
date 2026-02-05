import sys
import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox, Listbox, Scrollbar
from PIL import Image, ImageTk, ImageOps
from PIL.ExifTags import TAGS, GPSTAGS # <--- [CẬP NHẬT] Thêm GPSTAGS
import joblib

# --- FIX LỖI PILLOW 10.0+ ---
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
    PIL.Image.linear_gradient = PIL.Image.new
# ---------------------------------------

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from paddleocr import PaddleOCR
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from geopy.geocoders import Nominatim # <--- [MỚI] Thêm Geopy

# ==============================================================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==============================================================================
BASE_DIR = r"D:\LuanVan"
YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'models','Best_Medium_640_50e_10p_auto_20260119_1637.pt')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DET_PATH = os.path.join(MODEL_DIR, 'ch_PP-OCRv3_det_infer')
REC_PATH = os.path.join(MODEL_DIR, 'en_PP-OCRv4_rec_infer')
VIETOCR_WEIGHT_PATH = os.path.join(MODEL_DIR, 'vgg_transformer.pth')
CLASSIFIER_PATH = os.path.join(BASE_DIR, "model_phanloai_danhmuc", "best_optimized_classifier.pkl")

# ==============================================================================
# BACKEND V12: TÍCH HỢP GEOPY & CLEAN ADDRESS
# ==============================================================================
class OCRBackend:
    def __init__(self):
        print(">> Đang tải models (V12 - Geo Features)...")
        self.yolo = YOLO(YOLO_MODEL_PATH)
        
        self.paddle = PaddleOCR(
            det_model_dir=DET_PATH, rec_model_dir=REC_PATH,
            det_limit_side_len=2500, det_db_unclip_ratio=2.0, det_db_thresh=0.2,       
            det_db_box_thresh=0.4, use_angle_cls=False, lang='en', show_log=False, rec=False, use_gpu=True
        )
        
        config = Cfg.load_config_from_name('vgg_transformer')
        config['weights'] = VIETOCR_WEIGHT_PATH
        config['cnn']['pretrained'] = False
        config['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.vietocr = Predictor(config)

        self.classifier = None
        if os.path.exists(CLASSIFIER_PATH):
            try:
                self.classifier = joblib.load(CLASSIFIER_PATH)
                print(f"Đã load model phân loại.")
            except Exception as e:
                print(f"Lỗi load model phân loại: {e}")
        else:
            print(f"Không tìm thấy file model phân loại.")

        # Khởi tạo Geocoders
        self.geolocator = Nominatim(user_agent="luan_van_app_v12")

    # --- [HELPER] CHUYỂN ĐỔI DMS SANG DECIMAL ---
    def get_decimal_from_dms(self, dms, ref):
        try:
            # Ép kiểu float để tránh lỗi IFDRational của PIL
            degrees = float(dms[0])
            minutes = float(dms[1]) / 60.0
            seconds = float(dms[2]) / 3600.0
            decimal = degrees + minutes + seconds
            if ref in ['S', 'W']:
                decimal = -decimal
            return decimal
        except Exception:
            return 0.0

    # --- [HELPER] LỌC ĐỊA CHỈ RÚT GỌN ---
    # --- [CẬP NHẬT] LẤY ĐỊA CHỈ CHI TIẾT HƠN ---
    def get_clean_address(self, location):
        """Hàm lọc lấy: Số nhà + Đường, Phường, Quận, Thành phố"""
        if not location:
            return "Không tìm thấy địa chỉ bản đồ"
            
        addr = location.raw.get('address', {})
        parts = []

        # 1. [SỬA] Cố gắng lấy cả Số nhà VÀ Tên đường
        street_part = ""
        if 'house_number' in addr: 
            street_part += f"Số {addr['house_number']}"
        
        if 'road' in addr:
            if street_part: street_part += " " # Thêm khoảng trắng nếu đã có số nhà
            street_part += addr['road']
        
        if street_part:
            parts.append(street_part)
        # -------------------------------------------------
        
        # 2. Lấy Phường/Xã (Giữ nguyên)
        ward = addr.get('quarter') or addr.get('ward') or addr.get('village')
        if ward: parts.append(ward)

        # 3. Lấy Quận/Huyện (Giữ nguyên)
        district = addr.get('city_district') or addr.get('district') or addr.get('county') or addr.get('suburb')
        if district: parts.append(district)

        # 4. Lấy Tỉnh/Thành phố (Giữ nguyên)
        city = addr.get('city') or addr.get('state')
        if city: parts.append(city)

        return ", ".join(parts) if parts else location.address

    # --- [CHÍNH] LẤY GPS VÀ ĐỊA CHỈ TỪ ẢNH ---
    def get_geo_info(self, img_path):
        try:
            image = Image.open(img_path)
            exif_data = image._getexif()

            if not exif_data:
                return "Không có metadata", "Không xác định"

            gps_info = {}
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, tag_id)
                if tag_name == "GPSInfo":
                    for key in value.keys():
                        sub_tag = GPSTAGS.get(key, key)
                        gps_info[sub_tag] = value[key]
            
            if 'GPSLatitude' in gps_info and 'GPSLongitude' in gps_info:
                lat = self.get_decimal_from_dms(gps_info['GPSLatitude'], gps_info.get('GPSLatitudeRef', 'N'))
                lon = self.get_decimal_from_dms(gps_info['GPSLongitude'], gps_info.get('GPSLongitudeRef', 'E'))

                coords_str = f"{lat:.6f}, {lon:.6f}"
                
                # Gọi Geopy để lấy địa chỉ thực tế
                try:
                    location = self.geolocator.reverse(coords_str, timeout=5)
                    short_address = self.get_clean_address(location)
                except Exception as e:
                    short_address = "Lỗi kết nối bản đồ"
                    print(f"Geopy Error: {e}")

                return coords_str, short_address
            else:
                return "Không có GPS", "Không xác định"

        except Exception as e:
            print(f"Lỗi đọc EXIF: {e}")
            return "Lỗi đọc file", "Lỗi"

    # --- HÀM REGEX TRÍCH XUẤT THÔNG TIN TRÊN BIỂN HIỆU ---
    def extract_contact_info(self, detected_texts):
        info = {
            "phone": [],
            "email": [],
            "address": []
        }

        full_text_combined = " ".join(detected_texts)

        # 1. TÌM SĐT
        phone_pattern = r'(?:\+84|84|0)(?:3|5|7|8|9|2\d)\d{1,2}[.\s-]?\d{3,4}[.\s-]?\d{3,4}\b'
        phones = re.findall(phone_pattern, full_text_combined)
        info["phone"] = list(set([p.replace('.', '').replace('-', '').replace(' ', '') for p in phones]))

        # 2. TÌM EMAIL
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        emails = re.findall(email_pattern, full_text_combined)
        info["email"] = list(set(emails))

        # 3. LÀM SẠCH TEXT
        clean_lines = []
        for line in detected_texts:
            t = re.sub(phone_pattern, ' ', line)
            t = re.sub(email_pattern, ' ', t)
            t = re.sub(r'(?i)\b(SĐT|ĐT|Tel|Hotline|Phone|Zalo|Call)\b.*$', '', t) # Cắt đuôi SĐT
            if t.strip():
                clean_lines.append(t.strip())

        # 4. LOGIC TÌM VÀ LÀM SẠCH ĐỊA CHỈ
        address_candidates = []
        
        # Regex nhận diện dòng địa chỉ (chấp nhận cả lỗi OCR như lc, dc...)
        start_keywords = r'(?i)^(ĐC|Đ\/c|Địa chỉ|Add|Address|Số|No\.|Đường|Khu|Tổ|Ấp|Cơ sở|Chi nhánh|\d+\s|lc|dc|loc)'
        geo_keywords = r'(?i)(Phường|Xã|Thị trấn|P\s|P\.|X\.|Quận|Huyện|Q\s|Q\.|Thành phố|TP|T\.|Tỉnh|Việt Nam|Cần Thơ|Hà Nội|HCM)'

        i = 0
        while i < len(clean_lines):
            line = clean_lines[i]
            
            # Kiểm tra xem dòng này có tiềm năng là địa chỉ không
            is_start_valid = re.search(start_keywords, line)
            is_geo_valid = re.search(geo_keywords, line)
            
            if is_start_valid or is_geo_valid:
                current_address = line
                
                # --- Nối dòng (nếu địa chỉ rớt xuống dòng dưới) ---
                while i + 1 < len(clean_lines):
                    next_line = clean_lines[i+1]
                    # Điều kiện nối: Bắt đầu bằng đơn vị hành chính hoặc là tên viết hoa ngắn
                    continuation_keys = r'(?i)^(Phường|Xã|Thị trấn|P\.|X\.|Quận|Huyện|Thành phố|TP|T\.|Tỉnh|Việt Nam)'
                    is_continuation = re.search(continuation_keys, next_line)
                    is_short_name = len(next_line) > 0 and len(next_line) < 25 and next_line[0].isupper()
                    
                    if is_continuation or is_short_name:
                        current_address += ", " + next_line
                        i += 1 
                    else:
                        break 
                
                # --- [QUAN TRỌNG] BƯỚC XÓA TIỀN TỐ "ĐC" ---
                # Regex này sẽ tìm các từ khóa đầu dòng + bất kỳ dấu câu nào (:, ., ,, -, khoảng trắng)
                # Ví dụ: "ĐC: ", "lc.", "DC,", "Địa chỉ -" sẽ bị thay thế bằng rỗng
                clean_addr = re.sub(r'(?i)^(ĐC|Đ\/c|Địa chỉ|Address|Add|lc|dc|loc|đc)[.:,\-\s]*', '', current_address).strip()
                
                # Chỉ lấy nếu chuỗi còn lại đủ dài (tránh trường hợp chỉ còn lại rác)
                if len(clean_addr) > 6: 
                    address_candidates.append(clean_addr)
            
            i += 1

        info["address"] = list(set(address_candidates))
        return info

    def enhance_contrast(self, img):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    def binary_threshold(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 15)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    def detect_robust(self, img):
        h_orig, w_orig = img.shape[:2]
        scale_factor = 640 / max(h_orig, w_orig)
        if scale_factor >= 1.0: scale_factor = 1.0; img_small = img
        else:
            new_w = int(w_orig * scale_factor); new_h = int(h_orig * scale_factor)
            img_small = cv2.resize(img, (new_w, new_h))
        
        ocr_res = self.paddle.ocr(img_small, cls=False, det=True, rec=False)
        if not ocr_res or ocr_res[0] is None: return []

        final_boxes = []
        for box in ocr_res[0]:
            box_np = np.array(box, dtype="float32")
            if scale_factor < 1.0: box_np /= scale_factor 
            final_boxes.append(box_np)
        return final_boxes

    def fit_line_and_group(self, boxes):
        if not boxes: return []
        box_infos = []
        for i, b in enumerate(boxes):
            center = np.mean(b, axis=0); h = np.linalg.norm(b[0] - b[3])
            box_infos.append({'idx': i, 'c': center, 'h': h, 'box': b})

        box_infos.sort(key=lambda x: x['c'][0])
        used_indices = set(); lines = []

        for i in range(len(box_infos)):
            if i in used_indices: continue
            seed = box_infos[i]; current_line = [seed['box']]; used_indices.add(i)
            line_avg_y = seed['c'][1]; line_avg_h = seed['h']
            
            for j in range(i + 1, len(box_infos)):
                if j in used_indices: continue
                candidate = box_infos[j]
                max_h = max(candidate['h'], line_avg_h); min_h = min(candidate['h'], line_avg_h)
                
                if (min_h / max_h) < 0.6: continue 
                if abs(candidate['c'][1] - line_avg_y) > min_h * 0.5: continue
                
                last_box_center = np.mean(current_line[-1], axis=0)
                if (candidate['c'][0] - last_box_center[0]) > line_avg_h * 3.0: continue

                current_line.append(candidate['box']); used_indices.add(j)
                line_avg_y = (line_avg_y * len(current_line) + candidate['c'][1]) / (len(current_line) + 1)
                line_avg_h = (line_avg_h * len(current_line) + candidate['h']) / (len(current_line) + 1)
            lines.append(current_line)
            
        lines.sort(key=lambda ln: sum([np.mean(b, axis=0)[1] for b in ln]) / len(ln))
        return lines

    def get_regression_rectified_crop(self, image, line_boxes):
        if not line_boxes: return None, None
        all_pts = np.concatenate(line_boxes, axis=0)
        tops = np.array([b[0] for b in line_boxes] + [b[1] for b in line_boxes], dtype=np.float32)
        bottoms = np.array([b[3] for b in line_boxes] + [b[2] for b in line_boxes], dtype=np.float32)

        line_top = cv2.fitLine(tops, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        line_bot = cv2.fitLine(bottoms, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        [vx_t, vy_t, x0_t, y0_t] = line_top; [vx_b, vy_b, x0_b, y0_b] = line_bot
        min_x = np.min(all_pts[:, 0]); max_x = np.max(all_pts[:, 0])

        def get_y_safe(x, vx, vy, x0, y0):
            return y0 if abs(vx) < 1e-2 else y0 + (vy/vx) * (x - x0)

        tl_y = get_y_safe(min_x, vx_t, vy_t, x0_t, y0_t); tr_y = get_y_safe(max_x, vx_t, vy_t, x0_t, y0_t)
        bl_y = get_y_safe(min_x, vx_b, vy_b, x0_b, y0_b); br_y = get_y_safe(max_x, vx_b, vy_b, x0_b, y0_b)
        src_pts = np.array([[min_x, tl_y], [max_x, tr_y], [max_x, br_y], [min_x, bl_y]], dtype="float32")

        w_new = np.linalg.norm(src_pts[1] - src_pts[0])
        h_new = max(np.linalg.norm(src_pts[3] - src_pts[0]), np.linalg.norm(src_pts[2] - src_pts[1]))

        if h_new > 5000 or w_new > 5000 or h_new <= 0 or w_new <= 0: return None, None
        pad_h = int(h_new * 0.2); pad_w = int(w_new * 0.05)
        dst_w = int(w_new + pad_w*2); dst_h = int(h_new + pad_h*2)
        dst_pts = np.array([[pad_w, pad_h], [dst_w-pad_w, pad_h], [dst_w-pad_w, dst_h-pad_h], [pad_w, dst_h-pad_h]], dtype="float32")

        try:
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(image, M, (dst_w, dst_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
            return src_pts.astype(int), warped
        except: return None, None

    def process_full_pipeline(self, img_path):
        # 1. [CẬP NHẬT] Lấy thông tin GPS và Địa chỉ thực tế
        gps_coords, geo_address = self.get_geo_info(img_path)

        # --- [FIX LỖI XOAY ẢNH DO EXIF] ---
        try:
            # Dùng PIL mở ảnh để đọc được Orientation tag
            pil_image = Image.open(img_path)
            
            # Hàm này tự động xoay ảnh về đúng chiều dựa trên EXIF
            pil_image = ImageOps.exif_transpose(pil_image)
            
            # Chuyển đổi từ PIL (RGB) sang OpenCV (BGR) để các model bên dưới hoạt động
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            print(f"Lỗi khi xử lý EXIF rotation: {e}")
            # Fallback về cách cũ nếu lỗi
            stream = open(img_path, "rb"); bytes = bytearray(stream.read()); stream.close()
            frame = cv2.imdecode(np.asarray(bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        # ----------------------------------

        results_data = []

        # YOLO Detect
        yolo_res = self.yolo(frame, conf=0.5, verbose=False)
        if not yolo_res[0].boxes: return frame, [], "Không tìm thấy biển hiệu", {}, gps_coords, geo_address
        
        box = yolo_res[0].boxes[0]; x1, y1, x2, y2 = map(int, box.xyxy[0])
        sign_crop = frame[y1:y2, x1:x2]
        sign_enhanced = self.enhance_contrast(sign_crop)

        # Paddle OCR & Grouping
        raw_boxes = self.detect_robust(sign_enhanced)
        valid_boxes = [b for b in raw_boxes if (np.max(b[:,1])-np.min(b[:,1]) > 8)]
        lines = self.fit_line_and_group(valid_boxes)

        detected_texts = [] 
        for idx, line_boxes in enumerate(lines):
            try:
                box_visual, crop_img = self.get_regression_rectified_crop(sign_enhanced, line_boxes)
                if crop_img is None or crop_img.size == 0: continue

                h, w = crop_img.shape[:2]; target_h = 64; target_w = int(w * (target_h / h))
                if target_w <= 0: target_w = 32
                crop_img = cv2.resize(crop_img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

                candidates = []
                for fname, func in [("Original", lambda x: x), ("Binary", self.binary_threshold)]:
                    proc = func(crop_img)
                    pil_input = Image.fromarray(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB))
                    text, prob = self.vietocr.predict(pil_input, return_prob=True)
                    candidates.append({'name': fname, 'text': text, 'prob': prob, 'img': proc})
                
                best = max(candidates, key=lambda x: x['prob'])
                if best['prob'] > 0.4:
                    detected_texts.append(best['text'])
                    results_data.append({
                        'id': idx + 1,
                        'box_points': box_visual.reshape((-1, 1, 2)),
                        'straight_img': crop_img,
                        'final_img': best['img'],
                        'filter_name': best['name'],
                        'text': best['text'],
                        'conf': best['prob']
                    })
            except: pass

        # --- PHÂN LOẠI & TRÍCH XUẤT THÔNG TIN ---
        predicted_category = "Chưa xác định"
        extracted_info = {"phone": [], "email": [], "address": []}
        
        # Tạo danh sách text đơn thuần từ results_data
        detected_texts_list = [res['text'] for res in results_data] # <--- DÒNG MỚI QUAN TRỌNG

        if detected_texts:
            full_context_text = " ".join(detected_texts_list) # Vẫn dùng để phân loại
            
            if self.classifier:
                try:
                    predicted_category = str(self.classifier.predict([full_context_text])[0])
                except: predicted_category = "Lỗi Model"
            
            # GỌI HÀM TRÍCH XUẤT MỚI (Truyền List thay vì String)
            extracted_info = self.extract_contact_info(detected_texts_list)

        # Trả về thêm geo_address
        return sign_crop, results_data, predicted_category, extracted_info, gps_coords, geo_address

# ==============================================================================
# FRONTEND: GIAO DIỆN NGƯỜI DÙNG (TKINTER)
# ==============================================================================
class OCRInspectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Công cụ Debug OCR & Smart Extraction - V12 (Geopy Integration)")
        self.root.geometry("1400x800")
        
        self.backend = OCRBackend()
        self.current_sign_img = None
        self.current_results = []

        # --- PANEL TRÁI ---
        left_panel = tk.Frame(root, width=450, bg="#f0f0f0") 
        left_panel.pack_propagate(False) 
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        tk.Button(left_panel, text="Chọn Ảnh", command=self.load_image, 
                  height=2, bg="#4CAF50", fg="white", font=("Arial", 12, "bold")).pack(fill=tk.X, pady=5)
        
        self.lbl_category = tk.Label(left_panel, text="LOẠI HÌNH: ...", 
                                     font=("Arial", 16, "bold"), fg="#D32F2F", bg="#FFEBEE", pady=10)
        self.lbl_category.pack(fill=tk.X, pady=(5,5))

        # GROUP HIỂN THỊ THÔNG TIN
        info_frame = tk.LabelFrame(left_panel, text="Thông tin trích xuất & Vị trí", font=("Arial", 10, "bold"), bg="white")
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # [CẬP NHẬT] Hiển thị GPS và Địa chỉ thực
        self.lbl_gps = tk.Label(info_frame, text=" GPS: ...", anchor="w", justify="left", bg="white", fg="blue", wraplength=400)
        self.lbl_gps.pack(fill=tk.X, padx=5, pady=2)

        self.lbl_geo_addr = tk.Label(info_frame, text=" Vị trí ảnh: ...", anchor="w", justify="left", bg="white", fg="#00695C", wraplength=400, font=("Arial", 9, "bold"))
        self.lbl_geo_addr.pack(fill=tk.X, padx=5, pady=2)

        tk.Label(info_frame, text="--- Thông tin trên biển hiệu ---", bg="white", fg="#888").pack(fill=tk.X, pady=2)

        self.lbl_phone = tk.Label(info_frame, text=" SĐT: ...", anchor="w", justify="left", bg="white")
        self.lbl_phone.pack(fill=tk.X, padx=5, pady=2)

        self.lbl_addr = tk.Label(info_frame, text=" Đ/C Biển hiệu: ...", anchor="w", justify="left", bg="white", wraplength=400)
        self.lbl_addr.pack(fill=tk.X, padx=5, pady=2)

        self.lbl_email = tk.Label(info_frame, text=" Email: ...", anchor="w", justify="left", bg="white")
        self.lbl_email.pack(fill=tk.X, padx=5, pady=2)

        # LISTBOX
        tk.Label(left_panel, text="Danh sách dòng chữ:", bg="#f0f0f0", font=("Arial", 12, "bold")).pack(pady=(10,0))
        list_frame = tk.Frame(left_panel)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        scrollbar = Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox = Listbox(list_frame, font=("Consolas", 14), activestyle='none',
                               bg="white", fg="black", selectbackground="#0078D7", selectforeground="white",
                               yscrollcommand=scrollbar.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.listbox.yview)
        self.listbox.bind('<<ListboxSelect>>', self.on_select_line)

        # --- PANEL PHẢI ---
        right_panel = tk.Frame(root, bg="white")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        tk.Label(right_panel, text="[1] Vị trí trên biển hiệu", bg="white", fg="blue").pack(anchor="w")
        self.lbl_sign = tk.Label(right_panel, bg="#dddddd")
        self.lbl_sign.pack(pady=5)

        tk.Label(right_panel, text="[2] Sau khi cắt & xoay thẳng", bg="white", fg="blue").pack(anchor="w")
        self.lbl_straight = tk.Label(right_panel, bg="#dddddd")
        self.lbl_straight.pack(pady=5)

        tk.Label(right_panel, text="[3] Ảnh đã xử lý -> VietOCR Input", bg="white", fg="blue").pack(anchor="w")
        self.lbl_final = tk.Label(right_panel, bg="#dddddd")
        self.lbl_final.pack(pady=5)

        self.lbl_result = tk.Label(right_panel, text="KẾT QUẢ CHI TIẾT...", font=("Arial", 14, "bold"), fg="red", bg="white")
        self.lbl_result.pack(pady=10)

    def cv2_to_tk(self, cv_img, max_width=900, max_height=200):
        h, w = cv_img.shape[:2]
        scale = min(max_width/w, max_height/h)
        if scale < 1:
            new_w, new_h = int(w*scale), int(h*scale)
            cv_img = cv2.resize(cv_img, (new_w, new_h))
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(Image.fromarray(img_rgb))

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image", "*.jpg;*.png;*.jpeg")])
        if not path: return
        
        self.listbox.delete(0, tk.END)
        self.lbl_result.config(text="Đang xử lý...")
        self.lbl_category.config(text="LOẠI HÌNH: Đang phân tích...")
        
        # Reset labels
        self.lbl_gps.config(text=" GPS: ...")
        self.lbl_geo_addr.config(text=" Vị trí ảnh: ...")
        self.lbl_phone.config(text=" SĐT: ...")
        self.lbl_addr.config(text=" Đ/C Biển hiệu: ...")
        self.lbl_email.config(text=" Email: ...")
        self.root.update()

        try:
            # GỌI BACKEND MỚI (NHẬN 6 GIÁ TRỊ)
            sign_img, results, category, info, gps_coords, geo_addr = self.backend.process_full_pipeline(path)
            
            self.current_sign_img = sign_img
            self.current_results = results

            # Hiển thị ảnh
            self.tk_sign = self.cv2_to_tk(sign_img, max_height=300)
            self.lbl_sign.config(image=self.tk_sign)

            # CẬP NHẬT THÔNG TIN
            self.lbl_category.config(text=f"LOẠI HÌNH: {category.upper()}")
            
            # 1. Thông tin địa lý (Từ EXIF & Geopy)
            self.lbl_gps.config(text=f" GPS: {gps_coords}")
            self.lbl_geo_addr.config(text=f" Vị trí ảnh: {geo_addr}")
            
            # 2. Thông tin trích xuất (Từ OCR Regex)
            phone_txt = ", ".join(info['phone']) if info['phone'] else "Không tìm thấy"
            self.lbl_phone.config(text=f" SĐT: {phone_txt}")

            email_txt = ", ".join(info['email']) if info['email'] else "Không tìm thấy"
            self.lbl_email.config(text=f" Email: {email_txt}")

            sign_addr_txt = "\n".join(info['address']) if info['address'] else "Không tìm thấy"
            self.lbl_addr.config(text=f" Đ/C Biển hiệu: {sign_addr_txt}")

            if not results:
                self.listbox.insert(tk.END, "(Không tìm thấy chữ)")
            else:
                for res in results:
                    self.listbox.insert(tk.END, f"Dòng {res['id']}: {res['text']} ({res['conf']:.2f})")
                self.lbl_result.config(text="Đã xử lý xong.")

        except Exception as e:
            messagebox.showerror("Lỗi", str(e))
            print(e)

    def on_select_line(self, event):
        selection = self.listbox.curselection()
        if not selection: return
        data = self.current_results[selection[0]]

        viz_sign = self.current_sign_img.copy()
        rect = cv2.minAreaRect(data['box_points'])
        box = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(viz_sign, [box], 0, (0, 255, 0), 3) 
        
        self.tk_sign_update = self.cv2_to_tk(viz_sign, max_height=300)
        self.lbl_sign.config(image=self.tk_sign_update)
        self.tk_straight = self.cv2_to_tk(data['straight_img'], max_height=100)
        self.lbl_straight.config(image=self.tk_straight)
        self.tk_final = self.cv2_to_tk(data['final_img'], max_height=100)
        self.lbl_final.config(image=self.tk_final)
        self.lbl_result.config(text=f"TEXT: {data['text']}\nĐộ tin cậy: {data['conf']:.2f}")

if __name__ == "__main__":
    root = tk.Tk()
    app = OCRInspectorApp(root)
    root.mainloop()
