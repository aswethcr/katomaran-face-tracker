# Katomaran Face Tracker

A computer-vision pipeline for detecting people and faces, tracking people across video frames, recognizing persistent identities using face embeddings, and counting entry/exit events using a virtual counting line.

---

## Features

- Person detection using YOLO
- Face detection using YOLO face detection
- Face alignment using InsightFace
- ArcFace face embeddings using InsightFace
- Persistent face identification
- Multi-frame face recognition
- Automatic registration of unknown faces
- Multi-object person tracking using ByteTrack
- Entry/exit detection using a virtual counting line
- Occupancy calculation
- SQLite database for face and event storage
- Event logging
- Entry image capture
- Exit image capture
- Processed video generation
- Final event report generation
- Configurable detection and recognition thresholds
- Support for video files
- Support for RTSP input

---

## System Architecture

The pipeline works in the following stages:

```text
                    INPUT VIDEO
                         │
                         ▼
              ┌─────────────────────┐
              │   YOLO Person       │
              │     Detection       │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │      ByteTrack      │
              │   Person Tracking   │
              └──────────┬──────────┘
                         │
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
     ┌─────────────────┐   ┌─────────────────┐
     │  Face Detection │   │  Person Track   │
     │      YOLO       │   │    TRACK_x      │
     └────────┬────────┘   └────────┬────────┘
              │                     │
              ▼                     │
     ┌─────────────────┐             │
     │   InsightFace   │             │
     │ Face Alignment  │             │
     └────────┬────────┘             │
              │                     │
              ▼                     │
     ┌─────────────────┐             │
     │    ArcFace      │             │
     │    Embedding    │             │
     └────────┬────────┘             │
              │                     │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │ Face Recognition    │
              │ & Registration      │
              └──────────┬──────────┘
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
       Existing Face           Unknown Face
       FACE_xxxx               Multi-frame
                               Registration
              │                     │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │ Virtual Counting    │
              │       Line          │
              └──────────┬──────────┘
                         │
                 ┌───────┴───────┐
                 ▼               ▼
               ENTRY            EXIT
                 │               │
                 └───────┬───────┘
                         ▼
              ┌─────────────────────┐
              │ Occupancy Tracker   │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ SQLite Event DB     │
              └──────────┬──────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        Event Logs             Event Images
              │
              ▼
        Processed Video
```

---

## Project Structure

```text
katomaran-face-tracker/
│
├── data/
│   └── videos/
│       └── record_20250620_183903.mp4
│
├── database/
│   └── faces.db
│
├── logs/
│   ├── events.log
│   ├── entries/
│   │   └── YYYY-MM-DD/
│   │       └── entry images
│   │
│   └── exits/
│       └── YYYY-MM-DD/
│           └── exit images
│
├── models/
│   ├── yolov11n-face.pt
│   └── yolo11n.pt
│
├── output/
│   └── processed_video.mp4
│
├── src/
│   ├── __init__.py
│   ├── database.py
│   ├── detector.py
│   ├── event_manager.py
│   ├── face_utils.py
│   ├── logger.py
│   ├── occupancy.py
│   ├── pipeline.py
│   ├── recognition_manager.py
│   ├── recognizer.py
│   ├── registration.py
│   ├── tracker.py
│   └── video.py
│
├── app.py
├── config.json
├── generate_report.py
├── requirements.txt
│
├── test_events.py
├── test_face_detection.py
├── test_occupancy.py
├── test_recognition.py
├── test_tracking.py
└── test_video.py
```

---

## Main Components

### 1. YOLO Person Detection

The project uses a YOLO person detection model to detect people in each video frame.

The person detector is configured to detect class `0`, which represents the person class.

Example model:

```text
models/yolo11n.pt
```

Person detections are passed to the tracking system.

---

### 2. YOLO Face Detection

A dedicated YOLO face model is used to detect faces.

Example:

```text
models/yolov11n-face.pt
```

The detected face bounding boxes are used to associate faces with tracked people.

---

### 3. InsightFace

InsightFace is used for:

- Face detection
- Face landmark detection
- Face alignment
- ArcFace embedding generation

The application initializes:

```python
FaceAnalysis(
    name="buffalo_l"
)
```

The resulting ArcFace embedding is normalized before recognition.

---

### 4. ArcFace Face Recognition

Each detected face is converted into an embedding vector.

The normalized embedding is compared against the embeddings stored in the SQLite database using cosine similarity.

Example persistent identities:

```text
FACE_0001
FACE_0002
FACE_0003
```

Recognition is controlled using a similarity threshold.

For example:

```text
Recognition threshold = 0.60
```

If the similarity is above the recognition threshold, the stored identity is returned.

---

### 5. Multi-frame Recognition

The system does not rely only on one frame when registering a new person.

Multiple face embeddings can be collected for a tracked person.

For example:

```text
Frame 1 → embedding
Frame 2 → embedding
Frame 3 → embedding
```

The embeddings are averaged and normalized to create a more stable representation.

The number of required samples is configurable.

Example:

```text
Recognition samples = 3
```

This helps reduce registration errors caused by:

- Motion
- Temporary blur
- Slight pose changes
- Lighting changes
- Partial face visibility

---

### 6. ByteTrack Person Tracking

ByteTrack is used to maintain a persistent temporary tracking ID for each person.

Example:

```text
TRACK_1
TRACK_2
TRACK_3
```

A track ID represents a person being followed through the current video sequence.

The track ID and persistent face ID are separate concepts.

Example:

```text
TRACK_10 → FACE_0005
```

`TRACK_10` is the temporary tracking identity.

`FACE_0005` is the persistent face identity.

---

## Face-to-Person Association

The system associates a detected face with a tracked person.

The matching process uses:

1. Bounding-box IoU
2. Face center containment
3. Center distance relative to the person's bounding-box height

This allows the face recognition result to be associated with the correct person track.

---

## Entry and Exit Detection

A virtual horizontal counting line is used to determine when a tracked person enters or exits.

The line position is configured using:

```json
"line_y": 760
```

The actual value can be changed in `config.json`.

The event manager monitors the center Y-coordinate of the tracked person's bounding box.

When a person crosses the configured line, an event is generated.

Example:

```text
ENTRY | FACE_0005
```

or:

```text
EXIT | FACE_0008
```

---

## Recognition During Tracking

Face recognition and person tracking operate independently.

This is important because a person's face may not always be visible.

For example:

```text
Person detected
       │
       ▼
TRACK_10
       │
       ├── Face visible
       │      │
       │      ▼
       │  FACE_0005
       │
       └── Face not visible
              │
              ▼
          TRACK_10 continues
```

Therefore, a person can continue to be tracked even when:

- They turn away from the camera
- Their face becomes temporarily invisible
- The face detector misses a frame
- The person crosses the counting line while facing away

This allows entry/exit counting to be based on the **person track**, rather than requiring face visibility at the exact moment of crossing.

---

## Occupancy Tracking

The occupancy tracker maintains the current number of people inside the monitored area.

It records:

```text
Total entries
Total exits
Current occupancy
```

For example:

```text
Total entries : 5
Total exits   : 3
Occupancy     : 2
```

The occupancy value is calculated from the generated entry and exit events.

---

## Database

The application uses SQLite for persistent storage.

The database is located at:

```text
database/faces.db
```

The database stores face information and event information used by the application.

The database allows previously registered faces to be recognized in later processing sessions.

---

## Event Logging

Application events are written to:

```text
logs/events.log
```

Example:

```text
RECOGNIZED | TRACK_10 | FACE_0005 | existing=True | similarity=0.7604
```

Registration example:

```text
REGISTERED | TRACK_3 | FACE_0017 | similarity=0.5641
```

Entry example:

```text
ENTRY | FACE_0005 | track=10
```

Exit example:

```text
EXIT | FACE_0008 | track=19
```

---

## Entry and Exit Images

When an entry or exit event occurs, the application saves an image of the tracked person.

Entry images are stored under:

```text
logs/entries/
```

Exit images are stored under:

```text
logs/exits/
```

Images are organized by date.

Example:

```text
logs/
├── entries/
│   └── 2026-10-04/
│       └── 100400_580890_FACE_0005_entry.jpg
│
└── exits/
    └── 2026-10-04/
        └── 100510_983866_TRACK_35_exit.jpg
```

---

## Configuration

The application uses:

```text
config.json
```

Configuration includes:

### Input

- Video file path
- RTSP URL

### Output

- Processed video path
- Display settings

### Models

- YOLO face model
- YOLO person model
- Processing device

### Detection

- Detection confidence
- Frame skipping

### Recognition

- Recognition similarity threshold
- Registration threshold
- Required recognition samples

### Counting

- Counting line position
- Event cooldown

### Database

- SQLite database path

### Logging

- Event log path
- Entry image directory
- Exit image directory

---

## Example Configuration

```json
{
    "input": {
        "video_path": "data/videos/record_20250620_183903.mp4",
        "rtsp_url": ""
    },

    "output": {
        "video_path": "output/processed_video.mp4",
        "display": true
    },

    "models": {
        "yolo_face_model": "models/yolov11n-face.pt",
        "yolo_person_model": "models/yolo11n.pt",
        "device": "mps"
    },

    "detection": {
        "confidence": 0.5,
        "skip_frames": 0
    },

    "recognition": {
        "similarity_threshold": 0.60,
        "registration_threshold": 0.50,
        "samples_required": 3
    },

    "counting": {
        "line_y": 760,
        "cooldown_frames": 30
    }
}
```

Use the actual values from your own `config.json` if they differ.

---

## Installation

### 1. Clone or create the project

```bash
git clone <your-repository-url>
cd katomaran-face-tracker
```

If the project is already present locally, simply enter the project directory:

```bash
cd katomaran-face-tracker
```

---

### 2. Create a virtual environment

```bash
python3 -m venv venv
```

---

### 3. Activate the virtual environment

On macOS/Linux:

```bash
source venv/bin/activate
```

On Windows:

```powershell
venv\Scripts\activate
```

---

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Model Files

Place the required model files inside:

```text
models/
```

Expected models:

```text
models/yolov11n-face.pt
models/yolo11n.pt
```

The exact model filenames can be changed through `config.json`.

---

## Running the Application

Activate the virtual environment:

```bash
source venv/bin/activate
```

Then run:

```bash
python app.py
```

The application will:

1. Load the configuration.
2. Load the YOLO face detector.
3. Load the YOLO person detector.
4. Load InsightFace.
5. Initialize ByteTrack.
6. Open the configured input source.
7. Detect people.
8. Track people.
9. Detect and recognize faces.
10. Register unknown identities when appropriate.
11. Detect entry/exit events.
12. Update occupancy.
13. Save event images.
14. Store events in SQLite.
15. Generate the processed output video.

---

## RTSP Input

The application can also use an RTSP camera stream.

Configure the RTSP URL in:

```json
"input": {
    "video_path": "",
    "rtsp_url": "rtsp://..."
}
```

When `rtsp_url` is provided, the application uses the RTSP stream instead of the configured video file.

---

## Output

After processing, the application generates the processed video:

```text
output/processed_video.mp4
```

The output video contains information such as:

```text
TRACK_10 FACE_0005 0.76
```

and:

```text
Occupancy: 2
Entries: 5
Exits: 3
Unique visitors: 17
```

It also displays the configured counting line.

---

## Example Processing Output

A successful processing run can produce output similar to:

```text
======================================
FULL PIPELINE COMPLETED
======================================

Frames processed   : 1502
Recognized tracks  : 22
Unique visitors    : 17
Total entries      : 5
Total exits        : 3
Occupancy          : 2
Output             : output/processed_video.mp4
Event log          : logs/events.log

======================================
```

The exact values depend on the input video and database contents.

---

## Reports

The project includes:

```text
generate_report.py
```

Run:

```bash
python generate_report.py
```

The report generator can be used to summarize the stored event information.

---

## Testing

The project contains component-level test scripts.

### Occupancy

```bash
python test_occupancy.py
```

### Events

```bash
python test_events.py
```

### Face Detection

```bash
python test_face_detection.py
```

### Recognition

```bash
python test_recognition.py
```

### Tracking

```bash
python test_tracking.py
```

### Video

```bash
python test_video.py
```

---

## Important Concepts

### Track ID

A track ID identifies a person being tracked in the current video sequence.

Example:

```text
TRACK_10
```

Track IDs are temporary and are generated by the tracking system.

---

### Face ID

A face ID represents a persistent recognized identity.

Example:

```text
FACE_0005
```

Face IDs are stored in the database.

---

### Track ID vs Face ID

These are not the same thing.

Example:

```text
TRACK_10 → FACE_0005
```

A person can have a temporary track ID while the database maintains their persistent face identity.

---

## Recognition Thresholds

The application uses two important thresholds.

### Recognition Threshold

Example:

```text
0.60
```

If the similarity score is at or above this value, the face can be considered a match with an existing registered identity.

### Registration Threshold

Example:

```text
0.50
```

If an unknown person's multi-frame averaged embedding meets the registration criteria, the system can create a new persistent face identity.

These values should be tuned according to the camera, lighting, image quality, and application environment.

---

## Multi-frame Registration

Unknown people are not immediately registered from a single embedding when track-based recognition is being used.

The application collects several samples:

```text
Sample 1
Sample 2
Sample 3
...
```

The samples are combined into an averaged embedding.

This helps provide a more stable representation than relying on a single frame.

---

## Temporary Recognition State

Recognition samples are maintained per person track.

For example:

```text
TRACK_3
 ├── embedding 1
 ├── embedding 2
 └── embedding 3
```

Once the track has been recognized or registered, the result is stored for that track so that the application does not repeatedly register the same track.

---

## Frame Processing

The application can optionally skip frames during detection.

This is configured using:

```json
"skip_frames": 0
```

For example:

```text
skip_frames = 0
```

means detection runs on every frame.

A larger value can reduce computational load but may affect detection and recognition quality.

---

## Hardware Acceleration

The processing device can be configured in:

```json
"device": "mps"
```

For Apple Silicon Macs, `mps` can be used where supported.

Other environments may use:

```text
cpu
cuda
```

depending on the installed libraries and hardware.

---

## Directory Management

The application automatically creates required output directories when they do not already exist.

These include:

```text
database/
logs/
logs/entries/
logs/exits/
output/
```

---

## Development Files

The following are development/environment files and do not need to be included in the core project documentation:

```text
venv/
__pycache__/
src/__pycache__/
```

These should normally be excluded from version control.

---

## Recommended `.gitignore`

Create a `.gitignore` file containing:

```gitignore
# Python
__pycache__/
*.py[cod]
*.pyo

# Virtual environment
venv/
.venv/

# macOS
.DS_Store

# Generated database
database/*.db
database/*.sqlite
database/*.sqlite3

# Logs
logs/*.log

# Generated event images
logs/entries/
logs/exits/

# Generated output
output/*.mp4
output/*.avi
output/*.mov

# IDE
.vscode/
.idea/

# Python tooling
.pytest_cache/
.mypy_cache/
```

---

## Recommended Project Cleanup

The following files/folders are normally not required in the final source repository:

```text
venv/
__pycache__/
src/__pycache__/
src_backup/
```

The virtual environment should be recreated using:

```bash
python3 -m venv venv
```

and dependencies should be installed using:

```bash
pip install -r requirements.txt
```

Generated files such as databases, logs, event images, and processed videos can also be excluded from Git depending on the project requirements.

---

## Technologies Used

The project uses the following technologies:

- Python
- OpenCV
- NumPy
- Ultralytics YOLO
- InsightFace
- ArcFace
- ByteTrack
- Supervision
- SQLite

---

## Processing Flow Summary

```text
Video
  │
  ▼
YOLO Person Detection
  │
  ▼
ByteTrack
  │
  ▼
Person Track
  │
  ├─────────────── Face Detection
  │                      │
  │                      ▼
  │                 InsightFace
  │                      │
  │                      ▼
  │                 ArcFace
  │                      │
  │                      ▼
  │              Face Recognition
  │                      │
  │              ┌───────┴───────┐
  │              ▼               ▼
  │          Existing         Unknown
  │           FACE_ID       Registration
  │
  ▼
Counting Line
  │
  ├── ENTRY
  │
  └── EXIT
  │
  ▼
Occupancy
  │
  ▼
SQLite Database
  │
  ├── Events
  ├── Face IDs
  └── Timestamps
  │
  ▼
Logs + Event Images
  │
  ▼
Processed Video
```

---

## Limitations

Recognition performance depends on:

- Camera resolution
- Face visibility
- Lighting conditions
- Camera angle
- Motion blur
- Face pose
- Detection accuracy
- Recognition threshold configuration
- Quality of stored face embeddings

A person whose face is not visible may still be tracked for entry/exit purposes, but their persistent `FACE_xxxx` identity may not be available unless they were recognized earlier in the track.

---

## Future Improvements

Potential improvements include:

- Better face quality filtering
- Pose-aware recognition
- More robust track-to-face association
- Improved unknown-person handling
- Automatic threshold calibration
- GPU optimization
- Real-time RTSP optimization
- Web-based monitoring dashboard
- Live occupancy dashboard
- REST API
- Multiple camera support
- Cross-camera identity tracking
- Improved report generation
- Automatic database backup

---

## License

This project is a part of a hackathon run by https://katomaran.com