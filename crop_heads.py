import cv2
import os
from rembg import remove
from PIL import Image

# 輸入與輸出資料夾
input_folder = "input_images"
output_folder = "output_heads"
os.makedirs(output_folder, exist_ok=True)

# OpenCV 人臉偵測模型
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

for file in os.listdir(input_folder):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(input_folder, file)
        img = cv2.imread(img_path)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            margin_x = int(w * 0.4)
            margin_y = int(h * 0.8)
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(img.shape[1], x + w + margin_x)
            y2 = min(img.shape[0], y + h + margin_y)

            cropped = img[y1:y2, x1:x2]
            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(cropped_rgb)
            output_img = remove(pil_img)

            output_path = os.path.join(output_folder, os.path.splitext(file)[0] + ".png")
            output_img.save(output_path)
            print(f"✅ 處理完成: {file}")
        else:
            print(f"⚠️ 未偵測到人臉: {file}")

print("🎯 全部完成！輸出資料夾:", output_folder)
