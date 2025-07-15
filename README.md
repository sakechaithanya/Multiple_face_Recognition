
# 👥 Multiple‑Face‑Recognition (LBPH)

A lightweight Python application that **captures, trains, and recognizes multiple human faces in real‑time** using OpenCV’s **LBPH (Local Binary Patterns Histograms)** algorithm and Haar cascades.

---

## 📂 Project Structure

multiple_face_recognizer_lbph/
│
├─ capture_faces.py # Collect face images for each person
├─ train_model.py # Train LBPH model from the images
├─ recognize_faces.py # Real‑time face recognition demo
├─ dataset/ # Auto‑generated folders of captured images
│ └─ John_Doe/1.jpg ...
├─ trainer/
│ ├─ trainer.yml # Saved LBPH model (auto‑generated)
│ └─ name_id_map.json # Mapping between names and numeric IDs
└─ requirements.txt

## ⚙️ Requirements

| Package | Version (tested) | Why |
|---------|-----------------|-----|
| `opencv-contrib-python` | 4.10.x | Includes `cv2.face.*` (LBPH) |
| `numpy` | ≥1.20           | Array handling |
| `Pillow`| ≥10             | Image loading |
| *Optional*: `virtualenv`/`conda` for isolated envs |



🔍 How It Works
Stage	Details
Detection	       Haar cascade (haarcascade_frontalface_default.xml) locates faces in grayscale frames.
Capture	         capture_faces.py crops detected faces and saves them (default 50 samples per person).
Training	       train_model.py reads all images, converts them to uint8 arrays, assigns numeric IDs, and fits an LBPH recognizer.
Recognition	     recognize_faces.py loads the trained model, predicts each detected face, and labels it if confidence < 70.
