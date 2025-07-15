from flask import Flask, render_template, Response, request, jsonify
import cv2
import json
import os
import numpy as np
from PIL import Image
import threading
import time
from capture_faces import capture_faces
from train_model import train_model
import recognize_faces

app = Flask(__name__)

# Global variables for video capture
camera = None
frame = None
capture_thread = None
stop_event = threading.Event()

# Load name-ID mapping
name_id_map = {}
id_name_map = {}
try:
    with open("trainer/name_id_map.json", "r") as f:
        name_id_map = json.load(f)
    id_name_map = {v: k for k, v in name_id_map.items()}
except FileNotFoundError:
    pass

def generate_frames_recognize():
    global frame
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    try:
        recognizer.read("trainer/trainer.yml")
    except:
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + b'No trained model found.' + b'\r\n'
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while not stop_event.is_set():
        if frame is not None:
            # Use the recognize_faces function from recognize_faces.py
            processed_frame = recognize_faces.recognize_faces(frame, recognizer, face_cascade, id_name_map)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_data = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        time.sleep(0.1)

def generate_frames_capture():
    global frame
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    count = 0
    user_name = ""
    num_samples = 15
    save_path = ""

    while not stop_event.is_set():
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                count += 1
                face_img = gray[y:y+h, x:x+w]
                if save_path:
                    face_file = os.path.join(save_path, f"{count}.jpg")
                    cv2.imwrite(face_file, face_img)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"Sample {count}/{num_samples}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_data = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
            if count >= num_samples:
                break
        time.sleep(0.1)

def capture_camera():
    global camera, frame
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open camera.")
        return
    while not stop_event.is_set():
        ret, frame = camera.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        time.sleep(0.033)
    camera.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture', methods=['GET', 'POST'])
def capture():
    global capture_thread, stop_event, save_path
    if request.method == 'POST':
        user_name = request.form.get('user_name')
        if user_name:
            stop_event.clear()
            save_path = os.path.join("dataset", user_name.replace(" ", "_"))
            os.makedirs(save_path, exist_ok=True)
            capture_thread = threading.Thread(target=capture_camera)
            capture_thread.start()
            return jsonify({'status': 'started', 'user_name': user_name})
        return jsonify({'status': 'error', 'message': 'Name cannot be empty.'})
    return render_template('capture.html')

@app.route('/train')
def train():
    try:
        train_model()
        return render_template('train.html', message=" ")
    except Exception as e:
        return render_template('train.html', message=f"Error training model: {str(e)}")

@app.route('/recognize')
def recognize():
    global capture_thread, stop_event
    stop_event.clear()
    capture_thread = threading.Thread(target=capture_camera)
    capture_thread.start()
    return render_template('recognize.html')

@app.route('/video_feed_recognize')
def video_feed_recognize():
    return Response(generate_frames_recognize(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_capture')
def video_feed_capture():
    return Response(generate_frames_capture(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global stop_event, capture_thread
    stop_event.set()
    if capture_thread:
        capture_thread.join()
    return jsonify({'status': 'stopped'})

if __name__ == '__main__':
    app.run(debug=True)


    #I need you to integrate database to this project which should have 2 drop downs in datatbase page like 'Registered Students' and 'Recognized Students' 
    # and give a database option in nav bar and as well as dashboard and store data respectively and in capture page i need you to add a box like 'Enter Class of Students' which should have drop down like 'M.SC in AI', 'M.SC in CS','M.Tech in AI','M.Tech in CS'#