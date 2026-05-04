# 🚗 Smart Parking System (AI-Powered LPR)

ระบบจัดการที่จอดรถอัจฉริยะที่ใช้เทคโนโลยี **Computer Vision** ในการตรวจจับและอ่านป้ายทะเบียนรถยนต์อัตโนมัติ (License Plate Recognition - LPR) เพื่อบันทึกเวลาเข้า-ออก และคำนวณค่าจอดรถแบบ Real-time

---

## 🌟 Key Features

* **Automatic Plate Detection & OCR**: ใช้โมเดล **YOLO** (You Only Look Once) ในการตรวจจับตำแหน่งป้ายทะเบียน (Detection) และดึงข้อมูลตัวอักษร ตัวเลข รวมถึงจังหวัดจากป้ายทะเบียนได้อย่างแม่นยำ (OCR)
* **Advanced Image Preprocessing**: มีระบบปรับปรุงคุณภาพรูปภาพอัตโนมัติ เช่น การดัดภาพให้ตรง (Deskew & PCA Alignment), การปรับ Contrast (CLAHE) และการเพิ่มความคมชัด (Sharpening) เพื่อเพิ่มความแม่นยำสูงสุดก่อนส่งให้ OCR
* **Parking Logic & Fee Calculation**: ระบบคำนวณระยะเวลาการจอดและค่าบริการอัตโนมัติ โดยมีเงื่อนไขจอดฟรี 30 นาทีแรก และคิดค่าบริการรายนาทีในลำดับถัดไป
* **Real-time Dashboard & Updates**: แสดงสถานะรถเข้า-ออก และสรุปรายได้ประจำวันผ่าน **WebSocket** ทำให้ข้อมูลบนหน้าจออัปเดตทันทีโดยไม่ต้องกด Refresh
* **Database Management**: บันทึกประวัติการจอดรถ พร้อมเก็บข้อมูลรูปภาพหลักฐานในรูปแบบ Base64 ลงในระบบฐานข้อมูล **SQLite**

---

## 🛠 Tech Stack

* **Backend**: FastAPI (Python)
* **AI/ML/Computer Vision**: YOLO Model (Ultralytics), OpenCV, Scikit-learn (KMeans/PCA)
* **Database**: SQLite
* **Frontend Interface**: HTML, JavaScript (Tailwind CSS/Bootstrap)
* **Communication**: WebSocket & REST API

---

## 📂 Project Structure

```text
├── main.py              # ไฟล์หลักควบคุม API Endpoints และ Logic ทั้งหมดของระบบ
├── database.py          # ส่วนจัดการ Schema และการเชื่อมต่อฐานข้อมูล SQLite
├── templates/           # ไฟล์หน้ากาก UI สำหรับแสดงผล Dashboard
├── runs/                # โฟลเดอร์จัดเก็บโมเดล YOLO (Detection & OCR weights)
└── parking.db           # ฐานข้อมูล SQLite (สร้างอัตโนมัติเมื่อรันระบบ)




🚀 How it Works
Image Acquisition: ระบบรับภาพจากกล้องผ่าน API Endpoint /predict

Vehicle & Plate Detection: YOLO Model ทำหน้าที่ตรวจจับตำแหน่งป้ายทะเบียนและทำการ Crop ภาพเฉพาะส่วนนั้นออกมา

Image Enhancement: ภาพป้ายทะเบียนจะถูกทำ Preprocessing เพื่อลบนอยซ์ ปรับความสว่าง และดัดมุมเอียง

OCR Processing: YOLO Model (OCR) จะอ่านตัวเลข ตัวอักษร และจังหวัด จากนั้นนำข้อมูลไปเปรียบเทียบในฐานข้อมูล

Business Logic & Database:

หากเป็นรถเข้าใหม่ ระบบจะสร้าง Record ใหม่ (ENTRY)

หากเป็นรถเดิมที่เคยเข้ามาแล้ว ระบบจะอัปเดตเวลาออกและคำนวณค่าธรรมเนียม (EXIT)

Real-time Broadcast: ส่งข้อมูลสรุปผลไปยังหน้า Dashboard ของเจ้าหน้าที่ผ่าน WebSocket ทันที

⚙️ Installation & Usage
Clone the repository

Bash
git clone [https://github.com/yourusername/smart-parking-lpr.git](https://github.com/yourusername/smart-parking-lpr.git)
cd smart-parking-lpr
Install dependencies

Bash
pip install -r requirements.txt
Run the server

Bash
uvicorn main:app --reload
Access the Dashboard
เปิด Browser ไปที่ http://127.0.0.1:8000