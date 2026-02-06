import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Load FaceNet ----------
embedder = FaceNet()

# ---------- Load Haar Cascade SAFELY ----------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise IOError("Haar cascade not loaded")

# ---------- Paths ----------
KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_FILE = "attendance.csv"
THRESHOLD = 0.5

known_embeddings = []
known_ids = []
known_names = []

# ---------- Load Enrolled Students ----------
for folder in os.listdir(KNOWN_FACES_DIR):
    student_id, student_name = folder.split("_")
    folder_path = os.path.join(KNOWN_FACES_DIR, folder)

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (160, 160))
        img = np.expand_dims(img, axis=0)

        embedding = embedder.embeddings(img)
        known_embeddings.append(embedding[0])
        known_ids.append(student_id)
        known_names.append(student_name)

known_embeddings = np.array(known_embeddings)

# ---------- Create Attendance File ----------
if not os.path.exists(ATTENDANCE_FILE):
    pd.DataFrame(
        columns=["Student_ID", "Student_Name", "Time"]
    ).to_csv(ATTENDANCE_FILE, index=False)

# ---------- Webcam ----------
cap = cv2.VideoCapture(0)
marked_students = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (160, 160))
        face = np.expand_dims(face, axis=0)

        embedding = embedder.embeddings(face)
        similarities = cosine_similarity(embedding, known_embeddings)

        best_match = np.argmax(similarities)
        score = similarities[0][best_match]

        label = "Unknown"

        if score > THRESHOLD:
            sid = known_ids[best_match]
            sname = known_names[best_match]
            label = f"{sid} - {sname}"

            if sid not in marked_students:
                marked_students.add(sid)
                time = datetime.now().strftime("%H:%M:%S")
                df = pd.read_csv(ATTENDANCE_FILE)
                df.loc[len(df)] = [sid, sname, time]
                df.to_csv(ATTENDANCE_FILE, index=False)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
