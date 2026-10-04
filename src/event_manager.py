from datetime import datetime


class EventManager:
    """
    Person-based line crossing manager.

    IMPORTANT:
    Line crossing is determined from the PERSON TRACK,
    not from face visibility.

    This means a person can:
        - face the camera
        - get recognized as FACE_0001
        - turn around
        - become invisible to the face detector
        - continue across the counting line
        - still generate ENTRY/EXIT

    ENTRY:
        previous center was above line
        current center is on/below line

    EXIT:
        previous center was below line
        current center is on/above line
    """

    def __init__(
        self,
        line_y=760,
        cooldown_frames=75
    ):
        self.line_y = int(line_y)

        self.cooldown_frames = max(
            0,
            int(cooldown_frames)
        )

        # Previous Y position for PERSON TRACK.
        #
        # {
        #     track_id: previous_center_y
        # }
        self.previous_y = {}

        # Last event frame for PERSON TRACK.
        #
        # {
        #     track_id: frame_number
        # }
        self.last_event_frame = {}

        # Persistent face identity associated
        # with each person track.
        #
        # {
        #     track_id: "FACE_0001"
        # }
        self.track_to_face = {}

    # ========================================================
    # FACE ASSOCIATION
    # ========================================================

    def set_face_id(
        self,
        track_id,
        face_id
    ):
        """
        Associate a recognized FACE_xxxx with
        a person track.

        face_id may be None.
        """

        track_id = int(track_id)

        if face_id is not None:
            self.track_to_face[
                track_id
            ] = str(face_id)

    # ========================================================
    # LINE CROSSING
    # ========================================================

    def update(
        self,
        track_id,
        face_id,
        center_y,
        frame_number
    ):
        """
        Update one PERSON track.

        Face identity is optional.

        The crossing itself is always based on
        the person's movement.
        """

        track_id = int(track_id)
        center_y = int(center_y)
        frame_number = int(frame_number)

        # If a face ID is available, remember it.
        if face_id is not None:
            self.track_to_face[
                track_id
            ] = str(face_id)

        persistent_face_id = (
            self.track_to_face.get(
                track_id
            )
        )

        previous = self.previous_y.get(
            track_id
        )

        # Always update previous position.
        self.previous_y[
            track_id
        ] = center_y

        # First observation of this person.
        if previous is None:
            return None

        # Cooldown prevents repeated crossings.
        last_frame = (
            self.last_event_frame.get(
                track_id
            )
        )

        if (
            last_frame is not None
            and
            frame_number - last_frame
            <
            self.cooldown_frames
        ):
            return None

        event_type = None

        # ----------------------------------------------------
        # ENTRY
        # ----------------------------------------------------

        if (
            previous < self.line_y
            and
            center_y >= self.line_y
        ):
            event_type = "ENTRY"

        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------

        elif (
            previous > self.line_y
            and
            center_y <= self.line_y
        ):
            event_type = "EXIT"

        if event_type is None:
            return None

        self.last_event_frame[
            track_id
        ] = frame_number

        # If this person was never recognized,
        # still count the crossing.
        event_face_id = (
            persistent_face_id
            if persistent_face_id is not None
            else "UNKNOWN"
        )

        return {
            "event_type": event_type,

            "face_id": event_face_id,

            "track_id": track_id,

            "frame_number": frame_number,

            "timestamp": datetime.now().isoformat(),
        }

    # ========================================================
    # CLEANUP
    # ========================================================

    def forget_track(
        self,
        track_id
    ):
        """
        Remove stale person-track state.
        """

        track_id = int(track_id)

        self.previous_y.pop(
            track_id,
            None
        )

        self.last_event_frame.pop(
            track_id,
            None
        )

        self.track_to_face.pop(
            track_id,
            None
        )

    # Backwards-compatible method.
    def forget_face(
        self,
        face_id
    ):
        """
        Remove all tracks associated with
        a particular face ID.
        """

        if face_id is None:
            return

        face_id = str(face_id)

        tracks_to_remove = []

        for track_id, stored_face_id in (
            self.track_to_face.items()
        ):

            if stored_face_id == face_id:
                tracks_to_remove.append(
                    track_id
                )

        for track_id in tracks_to_remove:
            self.forget_track(
                track_id
            )