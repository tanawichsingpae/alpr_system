# 🚗 Smart Parking System (AI-Powered LPR)

ระบบจัดการที่จอดรถอัจฉริยะที่ใช้เทคโนโลยี **Computer Vision** ในการตรวจจับและอ่านป้ายทะเบียนรถยนต์อัตโนมัติ (License Plate Recognition - LPR) เพื่อบันทึกเวลาเข้า-ออก และคำนวณค่าจอดรถแบบ Real-time

---

## 🌟 Key Features

* **Automatic Plate Detection & OCR**
  ใช้โมเดล **YOLO (You Only Look Once)** ในการตรวจจับตำแหน่งป้ายทะเบียน (Detection) และดึงข้อมูลตัวอักษร ตัวเลข รวมถึงจังหวัดจากป้ายทะเบียนได้อย่างแม่นยำ (OCR)

* **Advanced Image Preprocessing**
  มีระบบปรับปรุงคุณภาพรูปภาพอัตโนมัติ เช่น:

  * Deskew & PCA Alignment
  * Contrast Enhancement (CLAHE)
  * Image Sharpening

* **Parking Logic & Fee Calculation**

  * ฟรี 30 นาทีแรก
  * คิดค่าบริการแบบรายนาทีหลังจากนั้น

* **Real-time Dashboard & Updates**
  แสดงข้อมูลแบบ Real-time ผ่าน **WebSocket**

* **Database Management**
  เก็บข้อมูลการจอด + รูปภาพ (Base64) ลง **SQLite**

---

## 🛠 Tech Stack

* **Backend**: FastAPI (Python)
* **AI/ML/Computer Vision**: YOLO, OpenCV, Scikit-learn
* **Database**: SQLite
* **Frontend**: HTML, JavaScript (Tailwind / Bootstrap)
* **Communication**: WebSocket & REST API

---

## 📂 Project Structure

```text
├── main.py
├── database.py
├── templates/
├── runs/
└── parking.db
```

---

## 🚀 How it Works

1. **Image Acquisition**
   รับภาพจากกล้องหรือ API (`/predict`)

2. **Vehicle & Plate Detection**
   ใช้ YOLO ตรวจจับและ crop ป้ายทะเบียน

3. **Image Enhancement**

   * Deskew / PCA Alignment
   * CLAHE
   * Sharpening

4. **OCR Processing**
   อ่านเลขทะเบียน + ตัวอักษร + จังหวัด

5. **Business Logic**

   * ENTRY → บันทึกข้อมูลรถ
   * EXIT → คำนวณเวลา + ค่าจอด

6. **Real-time Update**
   ส่งข้อมูลไป Dashboard ผ่าน WebSocket

---

## ⚙️ Installation & Usage

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/smart-parking-lpr.git
cd smart-parking-lpr
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Server

```bash
uvicorn main:app --reload
```

### 4. Access Dashboard

```
http://127.0.0.1:8000
```
