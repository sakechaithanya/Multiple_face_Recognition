import cv2
import numpy as np
from PIL import Image
import os
import json

def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    faces = []
    ids = []

    # Create name-to-ID mapping
    name_id_map = {}
    current_id = 1

    for user_folder in os.listdir('dataset'):
        path = os.path.join('dataset', user_folder)
        if not os.path.isdir(path):
            continue

        if user_folder not in name_id_map:
            name_id_map[user_folder] = current_id
            current_id += 1

        user_id = name_id_map[user_folder]

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            pil_image = Image.open(img_path).convert('L')
            image_np = np.array(pil_image, 'uint8')
            faces.append(image_np)
            ids.append(user_id)

    recognizer.train(faces, np.array(ids))
    os.makedirs("trainer", exist_ok=True)
    recognizer.save("trainer/trainer.yml")

    # Save name-ID mapping
    with open("trainer/name_id_map.json", "w") as f:
        json.dump(name_id_map, f)

    print("[INFO] Model trained and saved successfully.")
    print("[INFO] Name-ID mapping saved to trainer/name_id_map.json")

if __name__ == "__main__":
    train_model()
