import cv2
from insightface.app import FaceAnalysis

VIDEO_PATH = "data/videos/record_20250620_183903.mp4"
OUTPUT_PATH = "output/face_detection_test.mp4"


# --------------------------------------------------
# 1. Initialize InsightFace
# --------------------------------------------------

app = FaceAnalysis(name="buffalo_l")

# ctx_id=0 uses CPU for this test.
# We will optimize the pipeline for your M2 later.
app.prepare(ctx_id=0, det_size=(640, 640))


# --------------------------------------------------
# 2. Open video
# --------------------------------------------------

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")


fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Video opened successfully")
print("FPS:", fps)
print("Resolution:", width, "x", height)


# --------------------------------------------------
# 3. Create output video
# --------------------------------------------------

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)


# --------------------------------------------------
# 4. Process frames
# --------------------------------------------------

frame_count = 0
face_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # Detect faces
    faces = app.get(frame)

    # Draw every detected face
    for face in faces:

        bbox = face.bbox.astype(int)

        x1, y1, x2, y2 = bbox

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Confidence
        confidence = float(face.det_score)

        cv2.putText(
            frame,
            f"Face {confidence:.2f}",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    face_count += len(faces)

    # Write processed frame
    out.write(frame)

    # Display
    cv2.imshow("Katomaran - Face Detection", frame)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# --------------------------------------------------
# 5. Cleanup
# --------------------------------------------------

cap.release()
out.release()
cv2.destroyAllWindows()

print()
print("Processing complete")
print("Frames processed:", frame_count)
print("Total face detections:", face_count)
print("Output:", OUTPUT_PATH)