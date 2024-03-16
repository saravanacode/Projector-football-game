from ultralytics import YOLO
import cv2
import cvzone
import math

# Load the YOLO model
model = YOLO("best.pt")

# Define the class names
classNames = ["ball"]

# Open the video file
video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame
    frame = cv2.resize(frame, (640, 480))

    # Detect objects in the frame
    results = model(frame, stream=True, verbose=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # Calculate width and height of the bounding box
            w, h = x2 - x1, y2 - y1
            
            # Print size and location of the ball
            print(f"Size: {w}x{h}, Location: ({x1},{y1})")
            
            # Draw bounding box and confidence
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cvzone.putTextRect(frame, f'{classNames[int(box.cls[0])]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    # Show the frame
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

