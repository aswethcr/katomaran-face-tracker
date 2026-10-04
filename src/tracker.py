import numpy as np
import supervision as sv


class FaceTracker:
    """
    ByteTrack wrapper.

    This tracker is intentionally generic:
    it can track PERSON detections as well as any
    other bounding boxes supplied by the application.
    """

    def __init__(
        self,
        track_activation_threshold=0.25,
        lost_track_buffer=90,
        minimum_matching_threshold=0.8,
        minimum_consecutive_frames=1,
    ):
        print("Initializing ByteTrack...")

        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            minimum_consecutive_frames=minimum_consecutive_frames,
        )

        print("Tracker ready.")

    def update(self, detections):
        """
        Update ByteTrack.

        detections:
            [
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float
                }
            ]

        Returns:
            supervision.Detections
        """

        if not detections:

            empty = sv.Detections(
                xyxy=np.empty(
                    (0, 4),
                    dtype=np.float32
                ),
                confidence=np.empty(
                    (0,),
                    dtype=np.float32
                ),
                class_id=np.empty(
                    (0,),
                    dtype=np.int32
                ),
            )

            return self.tracker.update_with_detections(
                empty
            )

        xyxy = np.asarray(
            [
                d["bbox"]
                for d in detections
            ],
            dtype=np.float32
        )

        confidence = np.asarray(
            [
                d.get("confidence", 1.0)
                for d in detections
            ],
            dtype=np.float32
        )

        class_id = np.asarray(
            [
                d.get("class_id", 0)
                for d in detections
            ],
            dtype=np.int32
        )

        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )

        return self.tracker.update_with_detections(
            sv_detections
        )
