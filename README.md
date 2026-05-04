# 🚗 Smart Parking System (AI-Powered LPR)

ระบบจัดการที่จอดรถอัจฉริยะที่ใช้เทคโนโลยี **Computer Vision** ในการตรวจจับและอ่านป้ายทะเบียนรถยนต์อัตโนมัติ (License Plate Recognition - LPR) เพื่อบันทึกเวลาเข้า-ออก และคำนวณค่าจอดรถแบบ Real-time

---

## 🌟 Key Features

* **Automatic Plate Detection & OCR**: ใช้โมเดล **YOLO** (You Only Look Once) ในการตรวจจับตำแหน่งป้ายทะเบียน (Detection) และดึงข้อมูลตัวอักษร ตัวเลข รวมถึงจังหวัดจากป้ายทะเบียนได้อย่างแม่นยำ (OCR)
* **Advanced Image Preprocessing**: มีระบบปรับปรุงคุณภาพรูปภาพอัตโนมัติ เช่น การดัดภาพให้ตรง (Deskew & PCA Alignment), การปรับ Contrast (CLAHE) และการเพิ่มความคมชัด (Sharpening) เพื่อเพิ่มความแม่นยำสูงสุดก่อนส่งให้ OCR
* **Parking Logic & Fee Calculation**: ระบบคำนวณระยะเวลาการจอดและค่าบริการอัตโนมัติ โดยมีเงื่อนไขจอดฟรี 30 นาทีแรก และคิดค่าบริการรายนาทีในลำดับถัดไป[cite: 3]
* **Real-time Dashboard & Updates**: แสดงสถานะรถเข้า-ออก และสรุปรายได้ประจำวันผ่าน **WebSocket** ทำให้ข้อมูลบนหน้าจออัปเดตทันทีโดยไม่ต้องกด Refresh
* **Database Management**: บันทึกประวัติการจอดรถ พร้อมเก็บข้อมูลรูปภาพหลักฐานในรูปแบบ Base64 ลงในระบบฐานข้อมูล **SQLite**[cite: 1, 4]

---

## 🛠 Tech Stack

* **Backend**: FastAPI (Python)
* **AI/ML/Computer Vision**: YOLO Model (Ultralytics), OpenCV, Scikit-learn (KMeans/PCA)
* **Database**: SQLite[cite: 1, 4]
* **Frontend Interface**: HTML, JavaScript (Tailwind CSS/Bootstrap)
* **Communication**: WebSocket & REST API

---

## 📂 Project Structure
```text
├── main.py              # ไฟล์หลักควบคุม API Endpoints และ Logic ทั้งหมดของระบบ[cite: 2]
├── database.py          # ส่วนจัดการ Schema และการเชื่อมต่อฐานข้อมูล SQLite[cite: 1, 4]
├── templates/           # ไฟล์หน้ากาก UI สำหรับแสดงผล Dashboard[cite: 2, 3]
├── runs/                # โฟลเดอร์จัดเก็บโมเดล YOLO (Detection & OCR weights)[cite: 2, 3]
└── parking.db           # ฐานข้อมูล SQLite (สร้างอัตโนมัติเมื่อรันระบบ)[cite: 1, 2]


🚀 How it Worksกระบวนการทำงานของระบบแบ่งออกเป็นขั้นตอนหลัก ดังนี้:[cite: 2, 3]Image Acquisition: ระบบรับภาพจากกล้องหรือการอัปโหลดไฟล์ผ่าน API Endpoint /predict ในรูปแบบ Multi-part form data[cite: 2, 3]Vehicle & Plate Detection: ใช้ YOLO Model (Detection) วิเคราะห์ภาพเพื่อระบุตำแหน่งของป้ายทะเบียนรถยนต์ และทำการ Crop ภาพเฉพาะส่วนป้ายทะเบียนออกมา[cite: 2, 3]Image Enhancement: ปรับปรุงคุณภาพรูปภาพที่ Crop มาด้วยเทคนิคทาง Computer Vision เช่น การทำ Deskew, PCA Alignment เพื่อดัดภาพให้ตรง และใช้ CLAHE กับ Sharpening เพื่อเพิ่มความคมชัดของตัวอักษร[cite: 3]OCR Processing: YOLO Model (OCR) จะอ่านค่าตัวเลข ตัวอักษรหมวดทะเบียน และชื่อจังหวัด จากนั้นระบบจะทำการเปรียบเทียบข้อมูลในฐานข้อมูล[cite: 2, 3]Business Logic & Database:[cite: 2, 3]ENTRY: หากไม่พบข้อมูลรถในระบบ ระบบจะบันทึกข้อมูลใหม่พร้อมภาพถ่ายหลักฐานลง SQLite  EXIT: หากพบข้อมูลรถเดิม ระบบจะคำนวณระยะเวลาจอดและค่าธรรมเนียม พร้อมอัปเดตสถานะการออก[cite: 2, 3]Real-time Broadcast: ส่งข้อมูลเหตุการณ์และสรุปยอดรายได้ไปยังหน้า Dashboard ทันทีผ่าน WebSocket[cite: 2, 3]
