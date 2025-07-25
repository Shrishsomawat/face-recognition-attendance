import cv2
import numpy as np
from PIL import Image
import os

# Paths
dataset_path = 'dataset'
trainer_path = 'trainer'
if not os.path.exists(trainer_path):
    os.makedirs(trainer_path)

# LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    face_samples = []
    ids = []

    for image_path in image_paths:
        gray_img = Image.open(image_path).convert('L')  # Grayscale
        img_np = np.array(gray_img, 'uint8')

        # Filename format: userID_name_count.jpg → we extract ID
        user_id = int(os.path.split(image_path)[-1].split('_')[0])
        faces = detector.detectMultiScale(img_np)

        for (x, y, w, h) in faces:
            face_samples.append(img_np[y:y+h, x:x+w])
            ids.append(user_id)

    return face_samples, ids

print("⏳ Training faces. This may take a few seconds...")
faces, ids = get_images_and_labels(dataset_path)
recognizer.train(faces, np.array(ids))

# Save trained model
model_path = os.path.join(trainer_path, 'trainer.yml')
recognizer.save(model_path)
print(f"✅ Model trained and saved as {model_path}")
