import cv2
from datetime import datetime

from src.recognizer import FaceRecognizer
from src.database import FaceDatabase
from src.recognition_manager import RecognitionManager


VIDEO_PATH = "data/videos/record_20250620_183903.mp4"
OUTPUT_PATH = "output/recognition_test.mp4"


# --------------------------------------------------
# Initialize
# --------------------------------------------------

recognizer = FaceRecognizer()

database = FaceDatabase(
    "database/faces.db"
)

manager = RecognitionManager(
    database,
    similarity_threshold=0.45
)


# --------------------------------------------------
# Open video
# --------------------------------------------------

cap = cv2.VideoCapture(
    VIDEO_PATH
)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open video"
    )


fps = cap.get(
    cv2.CAP_PROP_FPS
)

width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)


# --------------------------------------------------
# Output
# --------------------------------------------------

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)


# --------------------------------------------------
# Track → Face mapping
# --------------------------------------------------

track_to_face = {}


# --------------------------------------------------
# Process
# --------------------------------------------------

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    faces = recognizer.get_faces(
        frame
    )

    for face in faces:

        embedding = recognizer.get_embedding(
            face
        )

        timestamp = datetime.now().isoformat()

        result = manager.identify_or_register(
            embedding,
            timestamp
        )

        face_id = result["face_id"]

        score = result["similarity"]

        x1, y1, x2, y2 = (
            face.bbox.astype(int)
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        label = (
            f"{face_id} "
            f"{score:.2f}"
        )

        cv2.putText(
            frame,
            label,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    out.write(frame)

    cv2.imshow(
        "Katomaran - Recognition",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
out.release()

cv2.destroyAllWindows()

database.close()

print()
print("Recognition test completed")
print(
    "Output:",
    OUTPUT_PATH
)