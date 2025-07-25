# 📸 Face Recognition Attendance System

A simple face recognition-based attendance system built using Python, OpenCV, and LBPH. This project captures user face images, trains a face recognizer, and then marks attendance in real-time with timestamp into a CSV file.

## 🧠 Overview

This was rebuilt from scratch to demonstrate real-time computer vision using OpenCV, with a clean modular structure:
- Step 1: Capture user face dataset
- Step 2: Train face recognition model using LBPH
- Step 3: Detect and recognize faces in live webcam feed and log attendance

Built to showcase computer vision + automation logic with minimum resources.

---

## 🔧 Tech Stack

- Python 3
- OpenCV (`opencv-contrib-python`)
- Haarcascade
- NumPy
- CSV File I/O
- LBPH Face Recognizer

---

## 📂 Folder Structure

face-recognition-attendance/
├── dataset/ # Captured face images
├── trainer/ # Trained model (trainer.yml)
├── 01_capture_dataset.py
├── 02_train_model.py
├── 03_face_recognizer.py
├── attendance.csv # Output file
├── requirements.txt
├── haarcascade_frontalface_default.xml
└── README.md


---

## 🚀 How to Run the Project

### 1. Install Dependencies

pip install -r requirements.txt
Step 1: Capture Your Face
python 01_capture_dataset.py
Enter your User ID and Name

Webcam will open and save 50 images in /dataset/

Step 2: Train the Model
python 02_train_model.py
Will train using saved images

Model saved as trainer/trainer.yml

Step 3: Run Face Recognition
python 03_face_recognizer.py
Webcam will open

Recognized faces will be marked in attendance.csv



Sample Output (CSV)
->ID, Name, DateTime
1, Shrish, 2025-07-25 22:33:05


📚 Learnings & Concepts
Haarcascade-based face detection using OpenCV

LBPH face recognition algorithm

Real-time face detection via webcam stream

Attendance logging using CSV

Dataset creation and training pipeline using NumPy + PIL

Folder organization and modular coding



💡 Potential Future Upgrades
GUI using Tkinter or PyQt5

Audio feedback ("Attendance marked!")

Face re-train/delete support

Push logs to Firebase/Supabase

Cloud-based deployment (Streamlit or Flask GUI)


👨‍💻 Developed By
Shrish Somawat
🚀 Python & Automation Enthusiast | Ex-Amazon (Tech-Ops)
🔗 LinkedIn • GitHub







---

## 🎯 After Adding README:
1. Commit the README changes on GitHub
2. Add `demo.gif` or screenshot in a `screenshots/` folder (optional but awesome)
3. Share your repo on LinkedIn with a short post — and I’ll help you draft that next 👇

Shall we do your **LinkedIn upgrade and project sharing post** next? Let’s make your comeback visible 🔥




