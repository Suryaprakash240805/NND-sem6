import cv2
import os

student_id = input("Enter Student ID: ")
student_name = input("Enter Student Name: ")

folder_name = f"{student_id}_{student_name}"
save_path = os.path.join("known_faces", folder_name)
os.makedirs(save_path, exist_ok=True)

# ✅ Correct Haar Cascade loading
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)
count = 0
MAX_IMAGES = 10

print("Press SPACE to capture image | ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Student Enrollment", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
    elif key == 32 and len(faces) > 0 and count < MAX_IMAGES:  # SPACE
        count += 1
        img_path = os.path.join(save_path, f"{count}.jpg")
        cv2.imwrite(img_path, face)
        print(f"Image {count} captured")

    if count >= MAX_IMAGES:
        print("Enrollment complete")
        break

cap.release()
cv2.destroyAllWindows()
