import os
import cv2
import json
import logging
import numpy as np

from datetime import datetime

from ultralytics import YOLO
from insightface.app import FaceAnalysis

from src.detector import FaceDetector
from src.tracker import FaceTracker
from src.database import FaceDatabase
from src.recognition_manager import RecognitionManager
from src.event_manager import EventManager
from src.occupancy import OccupancyTracker


# ============================================================
# CONFIGURATION
# ============================================================

CONFIG_PATH = "config.json"


def load_config(path):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with open(path, "r") as file:
        return json.load(file)


config = load_config(CONFIG_PATH)


VIDEO_PATH = config["input"]["video_path"]

RTSP_URL = config["input"].get(
    "rtsp_url",
    ""
)

OUTPUT_PATH = config["output"]["video_path"]

MODEL_PATH = config["models"]["yolo_face_model"]

PERSON_MODEL_PATH = config["models"].get(
    "yolo_person_model",
    "models/yolo11n.pt"
)

DATABASE_PATH = config["database"]["path"]

EVENT_LOG_PATH = config["logging"]["event_log"]

ENTRY_IMAGE_DIR = config["logging"]["entry_image_dir"]

EXIT_IMAGE_DIR = config["logging"]["exit_image_dir"]

DEVICE = config["models"].get(
    "device",
    "mps"
)

YOLO_CONFIDENCE = float(
    config["detection"]["confidence"]
)

DETECTION_SKIP = int(
    config["detection"]["skip_frames"]
)

RECOGNITION_THRESHOLD = float(
    config["recognition"]["similarity_threshold"]
)

REGISTRATION_THRESHOLD = float(
    config["recognition"]["registration_threshold"]
)

RECOGNITION_SAMPLES = int(
    config["recognition"]["samples_required"]
)

LINE_Y = int(
    config["counting"]["line_y"]
)

COOLDOWN_FRAMES = int(
    config["counting"]["cooldown_frames"]
)

DISPLAY = bool(
    config["output"].get(
        "display",
        True
    )
)


# ============================================================
# DIRECTORY SETUP
# ============================================================

for directory in [
    os.path.dirname(OUTPUT_PATH),
    os.path.dirname(DATABASE_PATH),
    ENTRY_IMAGE_DIR,
    EXIT_IMAGE_DIR,
    os.path.dirname(EVENT_LOG_PATH),
]:

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    filename=EVENT_LOG_PATH,
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    )
)

logger = logging.getLogger(
    "katomaran"
)


def log_event(message):

    print(message)

    logger.info(message)


# ============================================================
# HELPERS
# ============================================================

def clamp_bbox(
    bbox,
    width,
    height
):

    x1, y1, x2, y2 = bbox

    x1 = max(
        0,
        min(
            int(x1),
            width - 1
        )
    )

    y1 = max(
        0,
        min(
            int(y1),
            height - 1
        )
    )

    x2 = max(
        0,
        min(
            int(x2),
            width - 1
        )
    )

    y2 = max(
        0,
        min(
            int(y2),
            height - 1
        )
    )

    return (
        x1,
        y1,
        x2,
        y2
    )


def bbox_iou(
    box_a,
    box_b
):

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(
        ax1,
        bx1
    )

    iy1 = max(
        ay1,
        by1
    )

    ix2 = min(
        ax2,
        bx2
    )

    iy2 = min(
        ay2,
        by2
    )

    iw = max(
        0,
        ix2 - ix1
    )

    ih = max(
        0,
        iy2 - iy1
    )

    intersection = iw * ih

    if intersection <= 0:
        return 0.0

    area_a = (
        max(0, ax2 - ax1)
        *
        max(0, ay2 - ay1)
    )

    area_b = (
        max(0, bx2 - bx1)
        *
        max(0, by2 - by1)
    )

    union = (
        area_a
        +
        area_b
        -
        intersection
    )

    if union <= 0:
        return 0.0

    return (
        intersection / union
    )


def center_distance(
    box_a,
    box_b
):

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    acx = (ax1 + ax2) / 2
    acy = (ay1 + ay2) / 2

    bcx = (bx1 + bx2) / 2
    bcy = (by1 + by2) / 2

    return np.sqrt(
        (acx - bcx) ** 2
        +
        (acy - bcy) ** 2
    )


def normalize_embedding(
    embedding
):

    if embedding is None:
        return None

    embedding = np.asarray(
        embedding,
        dtype=np.float32
    ).reshape(-1)

    if embedding.size == 0:
        return None

    norm = np.linalg.norm(
        embedding
    )

    if not np.isfinite(norm):
        return None

    if norm < 1e-6:
        return None

    return (
        embedding / norm
    )


def get_face_embedding(
    face
):

    if face is None:
        return None

    return normalize_embedding(
        getattr(
            face,
            "embedding",
            None
        )
    )


def match_face_to_person(
    face_bbox,
    person_tracks
):

    """
    Match a detected face to a tracked person.
    """

    best_track = None
    best_score = -1.0

    fx1, fy1, fx2, fy2 = face_bbox

    fcx = (
        fx1 + fx2
    ) / 2

    fcy = (
        fy1 + fy2
    ) / 2

    for track in person_tracks:

        bbox = track["bbox"]

        px1, py1, px2, py2 = bbox

        iou = bbox_iou(
            face_bbox,
            bbox
        )

        inside = (
            px1 <= fcx <= px2
            and
            py1 <= fcy <= py2
        )

        distance = center_distance(
            face_bbox,
            bbox
        )

        person_height = max(
            1,
            py2 - py1
        )

        normalized_distance = (
            distance / person_height
        )

        score = iou

        if inside:
            score += 0.50

        score -= min(
            normalized_distance * 0.05,
            0.20
        )

        if score > best_score:

            best_score = score

            best_track = track

    return best_track


def save_event_image(
    frame,
    bbox,
    face_id,
    event_type
):

    """
    Save the complete person crop for the event.

    If a face is visible, the crop still contains the face.
    If the person is facing away, the person crop is still
    available for event evidence.
    """

    if face_id is None:
        face_id = "UNKNOWN"

    height, width = (
        frame.shape[:2]
    )

    x1, y1, x2, y2 = (
        clamp_bbox(
            bbox,
            width,
            height
        )
    )

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[
        y1:y2,
        x1:x2
    ]

    if crop.size == 0:
        return None

    now = datetime.now()

    date_folder = now.strftime(
        "%Y-%m-%d"
    )

    if event_type == "ENTRY":

        root = ENTRY_IMAGE_DIR

    else:

        root = EXIT_IMAGE_DIR

    directory = os.path.join(
        root,
        date_folder
    )

    os.makedirs(
        directory,
        exist_ok=True
    )

    filename = (
        f"{now.strftime('%H%M%S_%f')}_"
        f"{face_id}_"
        f"{event_type.lower()}.jpg"
    )

    path = os.path.join(
        directory,
        filename
    )

    cv2.imwrite(
        path,
        crop
    )

    return path


# ============================================================
# STARTUP
# ============================================================

print()
print("======================================")
print("KATOMARAN INTELLIGENT FACE TRACKER")
print("======================================")


# ============================================================
# FACE DETECTOR
# ============================================================

print()
print("Loading YOLO face detector...")

face_detector = FaceDetector(
    model_path=MODEL_PATH,
    device=DEVICE,
    confidence=YOLO_CONFIDENCE
)

print(
    f"YOLO face model : {MODEL_PATH}"
)

print(
    f"Device          : {DEVICE}"
)

print(
    f"Confidence      : {YOLO_CONFIDENCE}"
)

print(
    "YOLO face detector ready."
)


# ============================================================
# PERSON DETECTOR
# ============================================================

print()
print("Loading YOLO person detector...")

person_model = YOLO(
    PERSON_MODEL_PATH
)

print(
    f"Person model    : "
    f"{PERSON_MODEL_PATH}"
)

print(
    "YOLO person detector ready."
)


# ============================================================
# INSIGHTFACE
# ============================================================

print()
print("Loading InsightFace...")

insight_app = FaceAnalysis(
    name="buffalo_l"
)

insight_app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

print(
    "InsightFace ready."
)

print(
    "Using InsightFace detection + "
    "alignment + ArcFace embedding."
)


# ============================================================
# COMPONENTS
# ============================================================

print()
print("======================================")
print("Initializing components...")
print("======================================")


tracker = FaceTracker(
    track_activation_threshold=0.20,
    lost_track_buffer=120,
    minimum_matching_threshold=0.80,
    minimum_consecutive_frames=1
)


database = FaceDatabase(
    DATABASE_PATH
)


recognition = RecognitionManager(
    database,
    similarity_threshold=RECOGNITION_THRESHOLD,
    registration_threshold=REGISTRATION_THRESHOLD,
    samples_required=RECOGNITION_SAMPLES
)


events = EventManager(
    line_y=LINE_Y,
    cooldown_frames=COOLDOWN_FRAMES
)


occupancy = OccupancyTracker()


print(
    f"Recognition threshold   : "
    f"{RECOGNITION_THRESHOLD}"
)

print(
    f"Registration threshold  : "
    f"{REGISTRATION_THRESHOLD}"
)

print(
    f"Recognition samples     : "
    f"{RECOGNITION_SAMPLES}"
)

print(
    f"Detection skip          : "
    f"{DETECTION_SKIP}"
)

print(
    f"Counting line Y         : "
    f"{LINE_Y}"
)

print(
    "All components ready."
)


# ============================================================
# INPUT
# ============================================================

print()
print("======================================")
print("Opening input...")
print("======================================")


if RTSP_URL:

    input_source = RTSP_URL

    print(
        "Input type : RTSP"
    )

else:

    input_source = VIDEO_PATH

    print(
        "Input type : Video file"
    )

print(
    f"Input      : {input_source}"
)


cap = cv2.VideoCapture(
    input_source
)

if not cap.isOpened():

    database.close()

    raise RuntimeError(
        f"Could not open input: "
        f"{input_source}"
    )


fps = cap.get(
    cv2.CAP_PROP_FPS
)

if fps <= 0:
    fps = 25.0


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

total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)


print()
print("Video information:")

print(
    f"Resolution : "
    f"{width} x {height}"
)

print(
    f"FPS        : "
    f"{fps:.2f}"
)

print(
    f"Frames     : "
    f"{total_frames}"
)

print()


# ============================================================
# OUTPUT
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)

if not out.isOpened():

    cap.release()
    database.close()

    raise RuntimeError(
        "Could not create output video."
    )


# ============================================================
# STATE
# ============================================================

track_to_face = {}

track_last_embedding = {}

track_last_bbox = {}

track_face_similarity = {}

recognized_tracks = set()

unique_face_ids = set()

track_last_seen = {}

# ------------------------------------------------------------
# IMPORTANT
#
# Every person track gets a stable event identity.
#
# If a face is visible:
#     event identity = FACE_xxxx
#
# If face is NOT visible:
#     event identity = TRACK_x
#
# This prevents line counting from depending on face visibility.
# ------------------------------------------------------------

track_event_identity = {}

frame_number = 0

stopped_by_user = False

last_progress = -1


# ============================================================
# PROCESS VIDEO
# ============================================================

try:

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1


        # ====================================================
        # DETECTION
        # ====================================================

        run_detection = (
            DETECTION_SKIP <= 0
            or
            (
                (frame_number - 1)
                %
                (DETECTION_SKIP + 1)
                == 0
            )
        )


        person_detections = []

        insight_faces = []


        if run_detection:

            # ------------------------------------------------
            # PERSON DETECTION
            # ------------------------------------------------

            try:

                results = person_model.predict(
                    frame,
                    classes=[0],
                    conf=0.25,
                    verbose=False,
                    device=DEVICE
                )

                if results:

                    result = results[0]

                    boxes = result.boxes

                    if boxes is not None:

                        for box in boxes:

                            coords = (
                                box.xyxy[0]
                                .cpu()
                                .numpy()
                            )

                            confidence = float(
                                box.conf[0]
                                .cpu()
                                .item()
                            )

                            x1, y1, x2, y2 = (
                                clamp_bbox(
                                    coords,
                                    width,
                                    height
                                )
                            )

                            if (
                                x2 <= x1
                                or
                                y2 <= y1
                            ):
                                continue

                            person_detections.append({
                                "bbox": [
                                    x1,
                                    y1,
                                    x2,
                                    y2
                                ],
                                "confidence": confidence,
                                "class_id": 0
                            })

            except Exception as error:

                logger.exception(
                    "Person detection error: %s",
                    error
                )


            # ------------------------------------------------
            # FACE DETECTION
            # ------------------------------------------------

            try:

                # Keep YOLO face detector in the pipeline.
                face_detector.detect(
                    frame
                )

                # InsightFace provides:
                # - face detection
                # - alignment
                # - ArcFace embedding

                insight_faces = (
                    insight_app.get(
                        frame
                    )
                )

            except Exception as error:

                logger.exception(
                    "Face pipeline error: %s",
                    error
                )

                insight_faces = []


        # ====================================================
        # BYTE TRACK PERSONS
        # ====================================================

        tracked = tracker.update(
            person_detections
        )


        # ====================================================
        # BUILD CURRENT PERSON TRACKS
        # ====================================================

        current_tracks = []

        if (
            tracked.tracker_id is not None
            and
            len(tracked.xyxy) > 0
        ):

            for i in range(
                len(tracked.xyxy)
            ):

                tracker_id_value = (
                    tracked.tracker_id[i]
                )

                if tracker_id_value is None:
                    continue

                track_id = int(
                    tracker_id_value
                )

                x1, y1, x2, y2 = (
                    tracked.xyxy[i]
                    .astype(int)
                )

                x1, y1, x2, y2 = (
                    clamp_bbox(
                        [
                            x1,
                            y1,
                            x2,
                            y2
                        ],
                        width,
                        height
                    )
                )

                bbox = [
                    x1,
                    y1,
                    x2,
                    y2
                ]

                current_tracks.append({
                    "track_id": track_id,
                    "bbox": bbox
                })

                track_last_bbox[
                    track_id
                ] = bbox

                track_last_seen[
                    track_id
                ] = frame_number

                # --------------------------------------------
                # Create stable event identity immediately.
                #
                # This means even an unrecognized person can
                # cross the line and be counted.
                # --------------------------------------------

                if track_id not in track_event_identity:

                    track_event_identity[
                        track_id
                    ] = (
                        f"TRACK_{track_id}"
                    )


        # ====================================================
        # MATCH INSIGHTFACE FACES TO PERSON TRACKS
        # ====================================================

        if run_detection:

            for face in insight_faces:

                face_bbox = (
                    face.bbox.astype(int)
                )

                face_bbox = list(
                    clamp_bbox(
                        face_bbox,
                        width,
                        height
                    )
                )

                person_track = (
                    match_face_to_person(
                        face_bbox,
                        current_tracks
                    )
                )

                if person_track is None:
                    continue

                track_id = (
                    person_track[
                        "track_id"
                    ]
                )

                embedding = (
                    get_face_embedding(
                        face
                    )
                )

                if embedding is None:
                    continue

                track_last_embedding[
                    track_id
                ] = embedding


        # ====================================================
        # PROCESS PERSON TRACKS
        # ====================================================

        for track in current_tracks:

            track_id = (
                track["track_id"]
            )

            bbox = (
                track["bbox"]
            )

            x1, y1, x2, y2 = bbox


            # ------------------------------------------------
            # Recognition
            # ------------------------------------------------

            embedding = (
                track_last_embedding.get(
                    track_id
                )
            )

            face_id = (
                track_to_face.get(
                    track_id
                )
            )


            # ------------------------------------------------
            # IMPORTANT FIX:
            #
            # Pass track_id to RecognitionManager.
            #
            # This enables the configured multi-frame
            # recognition samples.
            # ------------------------------------------------

            if (
                face_id is None
                and
                embedding is not None
            ):

                try:

                    result = (
                        recognition.identify_or_register(
                            embedding,
                            datetime.now().isoformat(),
                            track_id=track_id
                        )
                    )

                except Exception as error:

                    logger.exception(
                        "Recognition error for "
                        "track %s: %s",
                        track_id,
                        error
                    )

                    result = {
                        "face_id": None,
                        "is_new": False,
                        "similarity": 0.0,
                        "valid": False,
                        "ready": False,
                        "samples": 0
                    }


                candidate_face_id = (
                    result.get(
                        "face_id"
                    )
                )

                similarity = float(
                    result.get(
                        "similarity",
                        0.0
                    )
                )

                is_new = bool(
                    result.get(
                        "is_new",
                        False
                    )
                )

                valid = bool(
                    result.get(
                        "valid",
                        False
                    )
                )

                sample_count = int(
                    result.get(
                        "samples",
                        0
                    )
                )


                # --------------------------------------------
                # Recognition successful
                # --------------------------------------------

                if (
                    valid
                    and
                    candidate_face_id
                ):

                    face_id = (
                        candidate_face_id
                    )

                    track_to_face[
                        track_id
                    ] = face_id

                    track_face_similarity[
                        track_id
                    ] = similarity

                    recognized_tracks.add(
                        track_id
                    )

                    unique_face_ids.add(
                        face_id
                    )


                    # ----------------------------------------
                    # IMPORTANT:
                    #
                    # Do NOT change the EventManager identity.
                    #
                    # It remains TRACK_x so the same person's
                    # previous line position remains connected
                    # even if recognition happens later.
                    # ----------------------------------------

                    if is_new:

                        log_event(
                            f"REGISTERED | "
                            f"TRACK_{track_id} | "
                            f"{face_id} | "
                            f"similarity="
                            f"{similarity:.4f} | "
                            f"samples="
                            f"{sample_count}"
                        )

                    else:

                        log_event(
                            f"RECOGNIZED | "
                            f"TRACK_{track_id} | "
                            f"{face_id} | "
                            f"existing=True | "
                            f"similarity="
                            f"{similarity:.4f} | "
                            f"samples="
                            f"{sample_count}"
                        )


            # ------------------------------------------------
            # CRITICAL COUNTING FIX
            #
            # ALWAYS use a stable event identity.
            #
            # This is independent from face visibility.
            # ------------------------------------------------

            event_identity = (
                track_event_identity.get(
                    track_id
                )
            )

            if event_identity is None:

                event_identity = (
                    f"TRACK_{track_id}"
                )

                track_event_identity[
                    track_id
                ] = event_identity


            # ------------------------------------------------
            # Center Y
            # ------------------------------------------------

            center_y = int(
                (y1 + y2) / 2
            )


            # ------------------------------------------------
            # ENTRY / EXIT
            #
            # IMPORTANT:
            #
            # Use the stable TRACK identity here.
            #
            # Therefore:
            #
            # Face visible:
            #     recognition still works
            #
            # Face not visible:
            #     person can STILL be counted
            # ------------------------------------------------

            event = events.update(
                track_id=track_id,
                face_id=event_identity,
                center_y=center_y,
                frame_number=frame_number
            )


            if event:

                event_type = (
                    event["event_type"]
                )

                timestamp = (
                    event["timestamp"]
                )

                # --------------------------------------------
                # For logging/database:
                #
                # Prefer actual FACE_xxxx when recognized.
                # Otherwise use TRACK_x as temporary identity.
                # --------------------------------------------

                logged_face_id = (
                    face_id
                    if face_id is not None
                    else event_identity
                )


                image_path = (
                    save_event_image(
                        frame,
                        bbox,
                        logged_face_id,
                        event_type
                    )
                )


                # --------------------------------------------
                # Occupancy
                # --------------------------------------------

                occupancy.process_event(
                    event_type
                )


                # --------------------------------------------
                # Database
                # --------------------------------------------

                try:

                    database.add_event(
                        face_id=logged_face_id,
                        event_type=event_type,
                        timestamp=timestamp,
                        image_path=image_path
                    )

                except Exception as error:

                    logger.exception(
                        "Database event error: %s",
                        error
                    )


                # --------------------------------------------
                # Event log
                # --------------------------------------------

                log_event(
                    f"{event_type} | "
                    f"{logged_face_id} | "
                    f"track={track_id} | "
                    f"timestamp={timestamp} | "
                    f"image={image_path}"
                )


            # ------------------------------------------------
            # DRAW PERSON TRACK
            # ------------------------------------------------

            if face_id is not None:

                similarity = (
                    track_face_similarity.get(
                        track_id,
                        0.0
                    )
                )

                label = (
                    f"TRACK_{track_id} "
                    f"{face_id} "
                    f"{similarity:.2f}"
                )

                box_color = (
                    0,
                    255,
                    0
                )

            else:

                label = (
                    f"TRACK_{track_id} "
                    f"PERSON"
                )

                box_color = (
                    0,
                    165,
                    255
                )


            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                2
            )

            cv2.putText(
                frame,
                label,
                (
                    x1,
                    max(
                        25,
                        y1 - 10
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                box_color,
                2
            )


        # ====================================================
        # COUNTING LINE
        # ====================================================

        cv2.line(
            frame,
            (0, LINE_Y),
            (width, LINE_Y),
            (255, 0, 0),
            3
        )


        cv2.putText(
            frame,
            "COUNTING LINE",
            (
                30,
                LINE_Y - 10
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 0, 0),
            2
        )


        # ====================================================
        # SUMMARY
        # ====================================================

        summary = (
            occupancy.get_summary()
        )


        cv2.putText(
            frame,
            (
                f"Occupancy: "
                f"{summary['current_occupancy']}"
            ),
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )


        cv2.putText(
            frame,
            (
                f"Entries: "
                f"{summary['total_entries']}"
            ),
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )


        cv2.putText(
            frame,
            (
                f"Exits: "
                f"{summary['total_exits']}"
            ),
            (30, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )


        cv2.putText(
            frame,
            (
                f"Unique visitors: "
                f"{len(unique_face_ids)}"
            ),
            (30, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )


        # ====================================================
        # PROGRESS
        # ====================================================

        if total_frames > 0:

            progress = (
                frame_number
                /
                total_frames
                *
                100
            )

            progress_int = (
                int(progress / 10)
                * 10
            )

            if progress_int != last_progress:

                print(
                    f"Processing: "
                    f"{progress_int}% "
                    f"({frame_number}/"
                    f"{total_frames})"
                )

                last_progress = (
                    progress_int
                )

        else:

            progress = 0


        cv2.putText(
            frame,
            (
                f"Progress: "
                f"{progress:.1f}%"
            ),
            (30, 195),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # ====================================================
        # WRITE OUTPUT
        # ====================================================

        out.write(
            frame
        )


        # ====================================================
        # DISPLAY
        # ====================================================

        if DISPLAY:

            cv2.imshow(
                "Katomaran - "
                "Person + Face Tracker",
                frame
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):

                stopped_by_user = True

                break


except KeyboardInterrupt:

    stopped_by_user = True

    print()
    print(
        "Processing interrupted."
    )


finally:

    cap.release()

    out.release()

    cv2.destroyAllWindows()

    database.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

final_summary = (
    occupancy.get_summary()
)


print()
print("======================================")


if stopped_by_user:

    print(
        "PROCESSING STOPPED BY USER"
    )

else:

    print(
        "FULL PIPELINE COMPLETED"
    )


print("======================================")


print(
    f"Frames processed   : "
    f"{frame_number}"
)

print(
    f"Recognized tracks  : "
    f"{len(recognized_tracks)}"
)

print(
    f"Unique visitors    : "
    f"{len(unique_face_ids)}"
)

print(
    f"Total entries      : "
    f"{final_summary['total_entries']}"
)

print(
    f"Total exits        : "
    f"{final_summary['total_exits']}"
)

print(
    f"Occupancy          : "
    f"{final_summary['current_occupancy']}"
)

print(
    f"Output             : "
    f"{OUTPUT_PATH}"
)

print(
    f"Event log          : "
    f"{EVENT_LOG_PATH}"
)

print("======================================")