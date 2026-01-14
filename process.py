# --- KIỂM TRA MÔI TRƯỜNG ---
import sys
try:
    import shapely
    if shapely.__version__.startswith("2."):
        print(f"ℹ️ Info: Đang dùng Shapely {shapely.__version__}")
except ImportError: pass

# --- FIX LỖI PILLOW ANTIALIAS ---
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
    PIL.Image.linear_gradient = PIL.Image.new

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from paddleocr import PaddleOCR
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import os
import traceback
import platform
# 🟢 THÊM THƯ VIỆN MATPLOTLIB ĐỂ HIỂN THỊ ZOOM
import matplotlib.pyplot as plt

# ==============================================================================
# 🔴 CẤU HÌNH ĐƯỜNG DẪN
# ==============================================================================
YOLO_MODEL_PATH = r'D:\LuanVan\BEST_MODEL_20260114_0257n.pt' 
IMAGE_PATH = r'Dataset_YOLO_Standard\val\images\images_95.jpg' 
MODEL_DIR = r'D:\LuanVan\models'

DET_PATH = os.path.join(MODEL_DIR, 'ch_PP-OCRv3_det_infer')
REC_PATH = os.path.join(MODEL_DIR, 'en_PP-OCRv4_rec_infer')
CLS_PATH = os.path.join(MODEL_DIR, 'ch_ppocr_mobile_v2.0_cls_infer')
VIETOCR_WEIGHT_PATH = os.path.join(MODEL_DIR, 'vgg_transformer.pth')
# ==============================================================================

class SignboardPipeline:
    def __init__(self, yolo_path):
        self.check_files()
        print("\n" + "="*40)
        print("⏳ Đang khởi tạo hệ thống...")
        
        self.yolo_model = YOLO(yolo_path)

        self.dbnet_detector = PaddleOCR(
            det_model_dir=DET_PATH,
            rec_model_dir=REC_PATH,
            cls_model_dir=CLS_PATH,
            use_angle_cls=True,
            lang='en', show_log=False, 
            rec=False, use_gpu=False
        ) 

        config = Cfg.load_config_from_name('vgg_transformer')
        config['weights'] = VIETOCR_WEIGHT_PATH 
        config['cnn']['pretrained'] = False
        config['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        config['predictor']['beamsearch'] = True 
        self.vietocr_recognizer = Predictor(config)
        
        print("✅ HỆ THỐNG SẴN SÀNG NHẬN DIỆN!")
        print("="*40 + "\n")

    def check_files(self):
        missing = []
        if not os.path.exists(YOLO_MODEL_PATH): missing.append(f"YOLO: {YOLO_MODEL_PATH}")
        if not os.path.exists(DET_PATH): missing.append(f"Paddle Det: {DET_PATH}")
        if not os.path.exists(VIETOCR_WEIGHT_PATH): missing.append(f"VietOCR: {VIETOCR_WEIGHT_PATH}")
        if missing:
            print("❌ THIẾU FILE MODEL:")
            for m in missing: print(f"   - {m}")
            sys.exit(1)

    def draw_vietnamese_text(self, img_cv, text, pos, font_size=20, color=(0, 0, 255)):
        try:
            img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            font_path = "arial.ttf" 
            if platform.system() == "Windows": font_path = "C:/Windows/Fonts/arial.ttf"
            try: font = ImageFont.truetype(font_path, font_size)
            except IOError: font = ImageFont.load_default()
            
            draw.text(pos, text, font=font, fill=color[::-1], stroke_width=2, stroke_fill=(255,255,255))
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except Exception: return img_cv

    def four_point_transform(self, image, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    def preprocess_ocr_input(self, img_crop):
        h, w = img_crop.shape[:2]
        if h < 32:
            scale = 32 / h
            img_crop = cv2.resize(img_crop, (int(w * scale), 32), interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB))

    def stack_images_vertically(self, img_list, bg_color=(255, 255, 255)):
        # Ghép ảnh trên nền trắng cho dễ nhìn
        if not img_list: return None
        max_w = max(img.shape[1] for img in img_list)
        total_h = sum(img.shape[0] for img in img_list) + (len(img_list) * 10) # Cách nhau 10px

        composite_img = np.full((total_h, max_w, 3), bg_color, dtype=np.uint8)

        current_y = 0
        for img in img_list:
            h, w = img.shape[:2]
            composite_img[current_y:current_y+h, 0:w] = img # Căn trái
            current_y += h + 10
        return composite_img

    # 🟢 HÀM HIỂN THỊ MỚI DÙNG MATPLOTLIB (HỖ TRỢ ZOOM)
    def show_zoomable_image(self, title, img_bgr):
        # Chuyển BGR sang RGB để hiển thị đúng màu
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        plt.figure(title, figsize=(10, 8)) # Tạo cửa sổ mới
        plt.title(title)
        plt.imshow(img_rgb)
        plt.axis('off') # Tắt trục tọa độ cho đẹp
        
        # Hiện thanh công cụ Zoom/Pan
        print(f"   👀 Đã mở cửa sổ: {title}. Hãy dùng công cụ Zoom (kính lúp) để soi chi tiết!")
        plt.show(block=False) # Không chặn chương trình, chạy tiếp

    def run(self, image_path):
        if not os.path.exists(image_path):
            print(f"❌ Không tìm thấy ảnh: {image_path}")
            return
        
        stream = open(image_path, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        frame = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
        stream.close()

        if frame is None:
            print("❌ Lỗi đọc ảnh.")
            return

        print(f"📄 Đang xử lý file: {os.path.basename(image_path)}")
        
        results = self.yolo_model(frame, conf=0.60, verbose=False) 
        if len(results[0].boxes) == 0:
            print("❌ Không tìm thấy biển hiệu nào.")
            return

        print(f"✅ Tìm thấy {len(results[0].boxes)} biển hiệu.")

        for i, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w_box, h_box = x2 - x1, y2 - y1
            
            if w_box * h_box < 1000: continue

            print(f"\n--- 🔍 Biển hiệu #{i+1} ---")
            sign_crop = frame[y1:y2, x1:x2]
            
            # 🟢 BƯỚC 1: HIỂN THỊ ẢNH YOLO CẮT (ZOOMABLE)
            self.show_zoomable_image(f"B1: Anh YOLO Cat (Bien {i+1})", sign_crop)

            # --- PADDLE OCR ---
            img_input = np.ascontiguousarray(sign_crop, dtype=np.uint8)
            try:
                result_ocr = self.dbnet_detector.ocr(img_input, cls=True, det=True, rec=False)
            except Exception as e:
                print(f"   ❌ Lỗi PaddleOCR: {e}")
                continue
            
            if not result_ocr or result_ocr[0] is None:
                print("   ⚠️ Không tìm thấy dòng chữ nào.")
                continue

            boxes = sorted(result_ocr[0], key=lambda b: b[0][1])
            results_text = []
            warped_crops_for_display = [] 

            for idx, txt_box in enumerate(boxes):
                try:
                    pts = np.array(txt_box, dtype="float32")
                    roi_warped = self.four_point_transform(sign_crop, pts)

                    h_roi, w_roi = roi_warped.shape[:2]
                    if h_roi < 5 or w_roi < 5: continue
                    
                    # Padding và thêm vào danh sách hiển thị
                    roi_padded = cv2.copyMakeBorder(roi_warped, 5, 5, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                    warped_crops_for_display.append(roi_padded)

                    # Nhận diện
                    roi_pil = self.preprocess_ocr_input(roi_padded)
                    text_pred = self.vietocr_recognizer.predict(roi_pil)
                    
                    print(f"   ▶ Dòng {idx+1}: {text_pred}")
                    results_text.append(text_pred)

                    # Vẽ kết quả
                    pts_int = pts.astype(int)
                    cv2.polylines(sign_crop, [pts_int], True, (0, 255, 0), 2)
                    sign_crop = self.draw_vietnamese_text(
                        sign_crop, 
                        text_pred, 
                        (pts_int[0][0], pts_int[0][1] - 25), 
                        font_size=20, 
                        color=(0, 0, 255)
                    )

                except Exception as e:
                    print(f"   ❌ Lỗi dòng {idx+1}: {e}")
            
            # 🟢 BƯỚC 2: HIỂN THỊ CÁC DÒNG CHỮ ĐÃ DUỖI (ZOOMABLE)
            if warped_crops_for_display:
                composite_warps = self.stack_images_vertically(warped_crops_for_display)
                if composite_warps is not None:
                    self.show_zoomable_image(f"B2: Chi tiet tung dong chu (Bien {i+1})", composite_warps)

            if results_text:
                full_text = " ".join(results_text)
                print(f"\n📝 KẾT QUẢ: \"{full_text}\"")

            # 🟢 BƯỚC 3: HIỂN THỊ KẾT QUẢ CUỐI (ZOOMABLE)
            self.show_zoomable_image(f"B3: KET QUA CUOI CUNG (Bien {i+1})", sign_crop)

        print("\n✅ Xử lý hoàn tất. Cửa sổ ảnh vẫn đang mở.")
        print("💡 Gợi ý: Dùng nút 'Kính lúp' trên cửa sổ ảnh để phóng to vùng chữ.")
        # Lệnh này giữ cửa sổ không bị tắt
        plt.show()

if __name__ == "__main__":
    try:
        pipeline = SignboardPipeline(YOLO_MODEL_PATH)
        pipeline.run(IMAGE_PATH)
    except Exception as e:
        print(f"\n❌ CHƯƠNG TRÌNH BỊ LỖI LỚN: {e}")
        traceback.print_exc()
        input("Nhấn Enter để thoát...")