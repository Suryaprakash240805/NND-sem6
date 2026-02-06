import cv2
from ultralytics import YOLO

# Load YOLOv8 Pose pretrained model
model = YOLO("yolov8n-pose.pt")

# Body part names for YOLOv8 keypoints
KEYPOINT_NAMES = [
    "Nose",
    "Left Eye", "Right Eye",
    "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder",
    "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist",
    "Left Hip", "Right Hip",
    "Left Knee", "Right Knee",
    "Left Ankle", "Right Ankle"
]

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO pose detection
    results = model(frame, conf=0.5)

    # Draw skeleton
    annotated_frame = results[0].plot()

    # Extract keypoints
    if results[0].keypoints is not None:
        keypoints = results[0].keypoints.xy.cpu().numpy()

        for person in keypoints:  # Loop through detected persons
            for i, (x, y) in enumerate(person):
                if x > 0 and y > 0:
                    cv2.putText(
                        annotated_frame,
                        KEYPOINT_NAMES[i],
                        (int(x), int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA
                    )

    # Show output
    cv2.imshow("YOLO Human Body Parts Detection", annotated_frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
