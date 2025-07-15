import cv2
import json

def recognize_faces(frame, recognizer, face_cascade, id_name_map):
    """
    Process a single frame for face recognition.
    Returns the frame with annotations.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    
    for (x, y, w, h) in faces:
        id, conf = recognizer.predict(gray[y:y+h, x:x+w])
        name = id_name_map.get(id, "Unknown") if conf < 70 else "Unknown"
        color = (0, 255, 0) if conf < 70 else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{name} ({int(conf)}%)", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return frame

# Only run standalone recognition if the script is executed directly
if __name__ == "__main__":
    print("[INFO] Starting face recognition. Press ESC to exit.")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    try:
        recognizer.read("trainer/trainer.yml")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    with open("trainer/name_id_map.json", "r") as f:
        name_id_map = json.load(f)
    id_name_map = {v: k for k, v in name_id_map.items()}
    
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break
        frame = recognize_faces(frame, recognizer, face_cascade, id_name_map)
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
            break
    
    cap.release()
    cv2.destroyAllWindows()