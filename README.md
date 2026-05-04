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

## 🚀 How it Works

กระบวนการทำงานของระบบแบ่งออกเป็น 5 ขั้นตอนหลัก ดังนี้:[cite: 2, 3]

1.  **Image Acquisition**: ระบบรับภาพจากกล้องหรือการอัปโหลดไฟล์ผ่าน API Endpoint `/predict` ในรูปแบบ Multi-part form data[cite: 2, 3]
2.  **Vehicle & Plate Detection**: ใช้ **YOLO Model** (Detection) วิเคราะห์ภาพเพื่อระบุตำแหน่งของป้ายทะเบียนรถยนต์ และทำการ Crop ภาพเฉพาะส่วนป้ายทะเบียนออกมาเพื่อเตรียมประมวลผลต่อ[cite: 2, 3]
3.  **Image Enhancement (Preprocessing)**: ปรับปรุงคุณภาพรูปภาพที่ Crop มาด้วยเทคนิคทาง Computer Vision:[cite: 3]
    *   **Deskew & PCA Alignment**: ดัดภาพที่เอียงให้ตรง[cite: 3]
    *   **CLAHE**: ปรับสมดุลแสงและความสว่างของภาพ[cite: 3]
    *   **Sharpening**: เพิ่มความคมชัดของตัวอักษรเพื่อลดความผิดพลาดของ OCR[cite: 3]
4.  **OCR Processing**: **YOLO Model (OCR)** จะอ่านค่าตัวเลข ตัวอักษรหมวดทะเบียน และชื่อจังหวัด จากนั้นระบบจะทำการเปรียบเทียบความคล้ายคลึง (Similarity Check) กับข้อมูลในฐานข้อมูล[cite: 2, 3]
5.  **Business Logic & Database**:[cite: 2, 3]
    *   **ENTRY**: หากไม่พบข้อมูลรถในระบบหรือเป็นรถที่ออกไปแล้ว ระบบจะบันทึกข้อมูลใหม่พร้อมภาพถ่ายหลักฐานลง SQLite[cite: 1, 2, 3]
    *   **EXIT**: หากพบข้อมูลรถที่ยังไม่ได้เช็คเอาท์ ระบบจะคำนวณระยะเวลาจอดและค่าธรรมเนียม พร้อมอัปเดตสถานะการออก[cite: 2, 3]
6.  **Real-time Broadcast**: ส่งข้อมูลเหตุการณ์ (Event) และสรุปยอดรายได้ประจำวันไปยังหน้า Dashboard ของเจ้าหน้าที่ทันทีผ่าน **WebSocket**[cite: 2, 3]

---

## ⚙️ Installation & Usage

ทำตามขั้นตอนด้านล่างนี้เพื่อติดตั้งและรันโปรเจคในเครื่องของคุณ:

### 1. Clone the repository
เปิด Terminal และรันคำสั่งเพื่อคัดลอกโปรเจค:
```bash
git clone [https://github.com/yourusername/smart-parking-lpr.git](https://github.com/yourusername/smart-parking-lpr.git)
cd smart-parking-lpr