import cv2
import os

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def capture_faces(user_name, num_samples=50):
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print(f"[INFO] Starting face capture for: {user_name}")
    count = 0
    save_path = os.path.join("dataset", user_name.replace(" ", "_"))
    os.makedirs(save_path, exist_ok=True)

    while True:
        ret, frame = cam.read()
        if not ret:
            print("[ERROR] Failed to capture frame from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            face_file = os.path.join(save_path, f"{count}.jpg")
            cv2.imwrite(face_file, face_img)

            # Draw rectangle and text
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"Sample {count}/{num_samples}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Show camera feed
        cv2.imshow("Face Capture - Press ESC to exit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or count >= num_samples:
            break

    print(f"[INFO] Finished capturing {count} images.")
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    user_name = input("Enter full name of the person: ").strip()
    if user_name:
        capture_faces(user_name)
    else:
        print("[ERROR] Name cannot be empty.")
