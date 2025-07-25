

import cv2
import os

# Load the pretrained Haar Cascade face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Input from user
user_id = input("Enter User ID: ")
user_name = input("Enter Name: ")

# Create dataset directory if not exists
dataset_path = "dataset"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

cam = cv2.VideoCapture(0)
sample_num = 0
max_samples = 50  # No. of images to capture

print("Capturing face. Look at the camera...")

while True:
    ret, frame = cam.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        sample_num += 1
        # Save the face image
        face_img = gray[y:y+h, x:x+w]
        file_path = os.path.join(dataset_path, f"{user_id}_{user_name}_{sample_num}.jpg")
        cv2.imwrite(file_path, face_img)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, f"Sample: {sample_num}/{max_samples}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    cv2.imshow("Face Capture", frame)

    if cv2.waitKey(1) == 27 or sample_num >= max_samples:  # Press ESC to exit
        break

cam.release()
cv2.destroyAllWindows()
print("Face capture completed.")
