# 🚗 Smart Parking System (AI-Powered LPR)

ระบบจัดการที่จอดรถอัจฉริยะที่ใช้เทคโนโลยี **Computer Vision** ในการตรวจจับและอ่านป้ายทะเบียนรถยนต์อัตโนมัติ (License Plate Recognition - LPR) เพื่อบันทึกเวลาเข้า-ออก และคำนวณค่าจอดรถแบบ Real-time

---

## 🌟 Key Features

- **Automatic Plate Detection & OCR**  
  ใช้โมเดล **YOLOv8** สำหรับตรวจจับป้ายทะเบียน และอ่านข้อมูลตัวอักษร ตัวเลข รวมถึงจังหวัดบนป้ายทะเบียน

- **Advanced Image Preprocessing**  
  ปรับปรุงคุณภาพภาพอัตโนมัติ เช่น:

  - Deskew  
  - PCA Alignment  
  - Image Upscaling  
  - Noise Reduction  
  - Contrast Enhancement  
  - Edge Sharpening  

- **Parking Logic & Fee Calculation**

  - ฟรี 30 นาทีแรก  
  - คิดค่าบริการแบบรายนาทีหลังจากนั้น  

- **Real-time Dashboard**

  - แสดงสถานะรถเข้า-ออกแบบ Real-time ผ่าน WebSocket  

- **Database Management**

  - เก็บข้อมูลรถเข้าออกและรูปภาพใน SQLite  

---

## 🛠 Tech Stack

- **Backend:** FastAPI (Python)  
- **AI/ML:** YOLOv8, OpenCV, Scikit-learn  
- **Database:** SQLite  
- **Frontend:** HTML, JavaScript, Tailwind CSS  
- **Communication:** REST API, WebSocket  

---

## 📂 Project Structure

```text
├── main.py
├── database.py
├── templates/
├── static/
├── runs/
└── parking.db
```

---

## 🔄 System Workflow

### Automatic License Plate Recognition Pipeline

ระบบรองรับการรับภาพจาก 2 ช่องทาง ได้แก่:

- กล้อง Real-time (Video Stream)
- อัปโหลดรูปภาพผ่านหน้าเว็บ

จากนั้นระบบจะประมวลผลตามขั้นตอนดังนี้:

### 1. Input Image

รับภาพเข้าสู่ระบบจากกล้องหรือการอัปโหลดไฟล์

### 2. Receive Image

ส่งภาพไปยัง FastAPI Endpoint เพื่อเริ่มกระบวนการตรวจจับ

### 3. Plate Detection

ใช้โมเดล **YOLOv8** ตรวจจับตำแหน่งป้ายทะเบียนจากภาพรถ

### 4. Select Best Bounding Box

เลือก Bounding Box ที่เหมาะสมที่สุดจากผลลัพธ์ที่ตรวจจับได้ โดยพิจารณาจาก:

- confidence score  
- object position  

📌 ตัวอย่างการเลือก Bounding Box:

```markdown
![Best Bounding Box](images\bounding_box.jpg)
```

### 5. Crop Plate Region

ตัดเฉพาะบริเวณป้ายทะเบียนออกจากภาพต้นฉบับ

### 6. Resize Plate Image

ปรับขนาดป้ายทะเบียนเป็น **320×160 pixels** เพื่อให้เหมาะกับการประมวลผล

### 7. Plate Preprocessing

ปรับปรุงคุณภาพภาพก่อน OCR:

- Deskew Plate  
- PCA Alignment  
- Image Upscaling  
- Noise Reduction  
- Contrast Enhancement  
- Edge Sharpening  

### 8. Character Detection

ใช้โมเดล OCR (**YOLOv8**) ตรวจจับตัวอักษรบนป้ายทะเบียน

📌 ตัวอย่าง Character Detection:

```markdown
![Character Detection](images\ocr.jpg)
```

### 9. OCR Fallback

หากโมเดลตรวจจับไม่ครบ ระบบจะใช้ OCR สำรองเพื่อช่วยอ่านข้อมูล

### 10. Remove Duplicate Characters

ลบ Bounding Box ที่ซ้อนกันหรือซ้ำซ้อน

### 11. Plate Parsing

จัดรูปแบบผลลัพธ์โดยแยก:

- ตัวอักษร  
- ตัวเลข  
- จังหวัด  

### 12. Output Result

ส่งผลลัพธ์กลับระบบ:

- หมายเลขทะเบียน  
- จังหวัด  
- ภาพป้ายทะเบียน  

---

## 🖼 System Pipeline Diagram

```markdown
![ALPR Pipeline](images\system_workflow.jpg)
```

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

### 4. Open Dashboard

```text
http://127.0.0.1:8000
```

---

## 📌 Core Functionality

- ตรวจจับป้ายทะเบียนจากภาพรถแบบอัตโนมัติ  
- อ่านเลขทะเบียนและจังหวัด  
- บันทึกเวลาเข้า-ออก  
- คำนวณค่าจอดรถ  
- อัปเดต Dashboard แบบ Real-time  
- จัดเก็บข้อมูลย้อนหลังในฐานข้อมูล  

```markdown
![Web Demo](images\web_demo.jpg)
```
---