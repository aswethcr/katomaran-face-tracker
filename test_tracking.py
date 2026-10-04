import cv2

from insightface.app import FaceAnalysis
from src.tracker import FaceTracker


VIDEO_PATH = "data/videos/record_20250620_183903.mp4"
OUTPUT_PATH = "output/tracking_test.mp4"


# --------------------------------------------------
# Face detector
# --------------------------------------------------

app = FaceAnalysis(name="buffalo_l")

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)


# --------------------------------------------------
# Video
# --------------------------------------------------

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Could not open video")


fps = cap.get(cv2.CAP_PROP_FPS)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)


# --------------------------------------------------
# Output
# --------------------------------------------------

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)


# --------------------------------------------------
# Tracker
# --------------------------------------------------

tracker = FaceTracker()


# --------------------------------------------------
# Processing
# --------------------------------------------------

frame_number = 0


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # Detect faces
    faces = app.get(frame)

    detections = []

    for face in faces:

        bbox = face.bbox.astype(int)

        x1, y1, x2, y2 = bbox

        detections.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": float(face.det_score)
        })


    # Track faces
    tracked = tracker.update(detections)


    # Draw tracked faces
    for i in range(len(tracked.xyxy)):

        x1, y1, x2, y2 = tracked.xyxy[i].astype(int)

        tracker_id = tracked.tracker_id[i]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"TRACK_{tracker_id}",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


    # Show frame
    cv2.imshow(
        "Katomaran - Face Tracking",
        frame
    )

    out.write(frame)


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
out.release()

cv2.destroyAllWindows()

print()
print("Tracking test completed")
print("Output:", OUTPUT_PATH)

