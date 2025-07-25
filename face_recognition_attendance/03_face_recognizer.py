import cv2
import numpy as np
import os
from datetime import datetime
import csv

# Load recognizer and face cascade
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load label names from dataset
def get_name_from_filename(filename):
    return "_".join(filename.split("_")[1:-1])

known_ids = {}
for file in os.listdir("dataset"):
    parts = file.split("_")
    if len(parts) >= 3:
        user_id = int(parts[0])
        user_name = parts[1]
        known_ids[user_id] = user_name

# Setup attendance file
def mark_attendance(user_id, name):
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")

    if not os.path.exists("attendance.csv"):
        with open("attendance.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Name", "DateTime"])

    # Avoid duplicate entries (basic)
    with open("attendance.csv", "r+") as f:
        data = f.read()
        if f"{user_id},{name}" not in data:
            writer = csv.writer(f)
            writer.writerow([user_id, name, dt_string])
            print(f"✅ Marked attendance: {user_id}, {name}")

# Start webcam
cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

print("🔍 Starting face recognition. Press ESC to quit.")

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        id_, confidence = recognizer.predict(face_img)

        if confidence < 60:
            name = known_ids.get(id_, "Unknown")
            mark_attendance(id_, name)
            label = f"{name} ({round(confidence, 2)}%)"
            color = (0, 255, 0)
        else:
            label = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y - 10), font, 0.8, color, 2)

    cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(1) == 27:  # ESC key
        break

cam.release()
cv2.destroyAllWindows()
