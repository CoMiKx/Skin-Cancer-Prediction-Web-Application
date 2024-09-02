from fastapi import FastAPI, File, UploadFile, HTTPException # นำเข้าโมดูลที่จำเป็นจาก FastAPI
from fastapi.responses import HTMLResponse # นำเข้า HTMLResponse สำหรับการส่งผลลัพธ์ในรูปแบบ HTML
from fastapi.staticfiles import StaticFiles # นำเข้า StaticFiles เพื่อให้บริการไฟล์ static
from fastapi.templating import Jinja2Templates # นำเข้า Jinja2Templates สำหรับการใช้งานเทมเพลต
import os # นำเข้า os เพื่อจัดการเส้นทางไฟล์
from tensorflow.keras.models import load_model # นำเข้า load_model จาก TensorFlow Keras เพื่อโหลดโมเดลที่บันทึกไว้
import uvicorn # นำเข้า uvicorn สำหรับการรันแอป FastAPI
import cv2 # นำเข้า OpenCV สำหรับจัดการภาพ
import numpy as np # นำเข้า NumPy สำหรับการประมวลผลข้อมูลภาพ
from PIL import Image # นำเข้า PIL สำหรับการจัดการภาพ
import io # นำเข้า io เพื่อจัดการข้อมูลไบนารี
from fastapi import Request # นำเข้า Request สำหรับจัดการการร้องขอ
import base64 # นำเข้า base64 สำหรับการเข้ารหัสภาพเป็น base64
import json # นำเข้า json สำหรับโหลดข้อมูล JSON

# โหลดโมเดลที่เทรนไว้ล่วงหน้า
model = load_model('my_skin_disease_pred_model.h5')

# สร้างแอปพลิเคชัน FastAPI
app = FastAPI()

# ตั้งค่าเส้นทางสำหรับเทมเพลตและไฟล์ static
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ฟังก์ชันสำหรับโหลดข้อมูลจากไฟล์ JSON
def load_data():
    with open(os.path.join("static", "data.json"), "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

# ฟังก์ชันสำหรับทำการพยากรณ์จากภาพที่อัพโหลด
def predict_image(image_data: bytes):
    # โหลดภาพโดยใช้ PIL
    image = Image.open(io.BytesIO(image_data))

    # แปลงภาพเป็น RGB หากยังไม่ได้อยู่ในโหมดนี้
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # แปลงภาพจาก PIL image เป็นอาร์เรย์ของ NumPy
    image = np.array(image)

    # เตรียมข้อมูลภาพ (เปลี่ยนขนาดและทำการ normalize)
    image = cv2.resize(image, (64, 64))  # เปลี่ยนขนาดภาพเป็น 64x64 ตามที่โมเดลคาดหวัง
    image = image.astype('float32') / 255.0  # ทำการ normalize ค่าพิกเซลให้อยู่ในช่วง [0, 1]

    # เพิ่มมิติ batch: (1, 64, 64, 3)
    image = np.expand_dims(image, axis=0)

    # ทำการพยากรณ์
    predictions = model.predict(image)

    # หาคลาสที่พยากรณ์ได้สูงสุดและค่าความน่าจะเป็น
    predicted_class_index = np.argmax(predictions[0])
    predicted_prob = predictions[0][predicted_class_index]
    predicted_class_name = data['class_names']['en'][f"{predicted_class_index}"]

    return predicted_class_name, predicted_prob, predicted_class_index

# เส้นทางสำหรับหน้าแรกของแอปพลิเคชัน
@app.get("/", response_class=HTMLResponse)
async def main_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# เส้นทางสำหรับทำการพยากรณ์เมื่ออัพโหลดภาพ
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    # ตรวจสอบประเภทไฟล์ ต้องเป็น JPEG หรือ PNG เท่านั้น
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG are supported.")

    # อ่านข้อมูลภาพ
    image_data = await file.read()

    # ทำการพยากรณ์จากภาพที่ได้รับ
    predicted_class_name, predicted_prob, predicted_class_index = predict_image(image_data)
    
    # แปลงภาพเป็น base64
    image = Image.open(io.BytesIO(image_data))
    target_height = 400
    image = image.resize((int(target_height * (image.width / image.height)), target_height))
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # แสดงผลลัพธ์ในเทมเพลต result.html
    return templates.TemplateResponse(
        "result.html", 
        {
            "request": request, 
            "picture": img_str, 
            "predicted_class_index": predicted_class_index, 
            "predicted_prob": f"{predicted_prob:.2%}",
        }
    )

if __name__ == '__main__':
    # โหลดข้อมูลจาก JSON
    data = load_data()
    uvicorn.run(app, host='0.0.0.0', port=8000) # รันแอปพลิเคชันบนพอร์ต 8000
