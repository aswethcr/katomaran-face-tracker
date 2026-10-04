import numpy as np


class RecognitionManager:
    """
    Manages face recognition and registration.

    Flow:
        1. Normalize incoming ArcFace embedding.
        2. Compare against all stored face embeddings.
        3. If similarity >= recognition_threshold:
              return existing FACE_xxxx
        4. Otherwise collect multiple samples for the track.
        5. After enough samples, register a new identity.

    Important:
        The embedding passed to this class must already be generated
        by InsightFace's ArcFace recognition model.
    """

    def __init__(
        self,
        database,
        similarity_threshold=0.60,
        registration_threshold=0.50,
        samples_required=3,
        max_samples=5,
    ):
        self.database = database

        self.similarity_threshold = float(
            similarity_threshold
        )

        self.registration_threshold = float(
            registration_threshold
        )

        self.samples_required = max(
            1,
            int(samples_required)
        )

        self.max_samples = max(
            self.samples_required,
            int(max_samples)
        )

        # Track-specific temporary recognition samples.
        #
        # {
        #     track_id: [embedding1, embedding2, ...]
        # }
        self.track_samples = {}

        # Prevent repeatedly registering the same track.
        self.track_results = {}

    # ========================================================
    # EMBEDDING HELPERS
    # ========================================================

    @staticmethod
    def normalize_embedding(embedding):
        """
        Convert embedding to float32 and L2-normalize it.
        """

        if embedding is None:
            return None

        try:
            embedding = np.asarray(
                embedding,
                dtype=np.float32
            ).reshape(-1)
        except Exception:
            return None

        if embedding.size == 0:
            return None

        if not np.all(
            np.isfinite(embedding)
        ):
            return None

        norm = np.linalg.norm(
            embedding
        )

        if not np.isfinite(norm):
            return None

        if norm < 1e-6:
            return None

        return embedding / norm

    @staticmethod
    def cosine_similarity(a, b):
        """
        Calculate cosine similarity.
        """

        a = RecognitionManager.normalize_embedding(
            a
        )

        b = RecognitionManager.normalize_embedding(
            b
        )

        if a is None or b is None:
            return 0.0

        if a.shape != b.shape:
            return 0.0

        score = float(
            np.dot(a, b)
        )

        if not np.isfinite(score):
            return 0.0

        return score

    # ========================================================
    # DATABASE MATCHING
    # ========================================================

    def identify(self, embedding):
        """
        Compare an embedding against all registered faces.

        Returns:
            (face_id, similarity)

        If no sufficiently strong match exists:
            (None, best_similarity)
        """

        embedding = self.normalize_embedding(
            embedding
        )

        if embedding is None:
            return None, 0.0

        faces = self.database.get_all_faces()

        best_face_id = None
        best_score = -1.0

        for face in faces:

            stored_embedding = (
                self.normalize_embedding(
                    face.get("embedding")
                )
            )

            if stored_embedding is None:
                continue

            score = self.cosine_similarity(
                embedding,
                stored_embedding
            )

            if score > best_score:
                best_score = score
                best_face_id = face.get(
                    "face_id"
                )

        if (
            best_face_id is not None
            and best_score >= self.similarity_threshold
        ):
            return (
                best_face_id,
                float(best_score)
            )

        return (
            None,
            max(
                float(best_score),
                0.0
            )
        )

    # ========================================================
    # REGISTER
    # ========================================================

    def register(
        self,
        embedding,
        timestamp
    ):
        """
        Register a new face in the database.
        """

        embedding = self.normalize_embedding(
            embedding
        )

        if embedding is None:
            return None

        existing = self.database.get_all_faces()

        # Avoid ID collision if records have been deleted.
        used_numbers = []

        for face in existing:

            face_id = str(
                face.get("face_id", "")
            )

            if face_id.startswith("FACE_"):

                try:
                    number = int(
                        face_id.split("_")[1]
                    )

                    used_numbers.append(
                        number
                    )

                except (
                    ValueError,
                    IndexError
                ):
                    pass

        if used_numbers:
            number = max(
                used_numbers
            ) + 1
        else:
            number = 1

        face_id = (
            f"FACE_{number:04d}"
        )

        self.database.add_face(
            face_id,
            embedding,
            timestamp
        )

        return face_id

    # ========================================================
    # SAMPLE MANAGEMENT
    # ========================================================

    def add_sample(
        self,
        track_id,
        embedding
    ):
        """
        Add one valid embedding to a track's
        temporary recognition sample buffer.

        Returns number of samples currently stored.
        """

        embedding = self.normalize_embedding(
            embedding
        )

        if embedding is None:
            return 0

        track_id = int(track_id)

        samples = self.track_samples.setdefault(
            track_id,
            []
        )

        # Avoid excessive duplicate samples.
        if samples:

            similarity = self.cosine_similarity(
                embedding,
                samples[-1]
            )

            # If the current frame is almost identical
            # to the previous sample, don't keep adding
            # the same embedding.
            if similarity > 0.995:
                return len(samples)

        samples.append(
            embedding
        )

        if len(samples) > self.max_samples:
            samples.pop(0)

        return len(samples)

    def get_samples(
        self,
        track_id
    ):
        """
        Return current samples for a track.
        """

        return self.track_samples.get(
            int(track_id),
            []
        )

    def clear_track(
        self,
        track_id
    ):
        """
        Clear temporary samples for a track.
        """

        track_id = int(track_id)

        self.track_samples.pop(
            track_id,
            None
        )

        self.track_results.pop(
            track_id,
            None
        )

    # ========================================================
    # MULTI-SAMPLE EMBEDDING
    # ========================================================

    def average_embedding(
        self,
        embeddings
    ):
        """
        Create a normalized average embedding.
        """

        if not embeddings:
            return None

        valid = []

        for embedding in embeddings:

            normalized = (
                self.normalize_embedding(
                    embedding
                )
            )

            if normalized is not None:
                valid.append(
                    normalized
                )

        if not valid:
            return None

        average = np.mean(
            np.stack(valid),
            axis=0
        )

        return self.normalize_embedding(
            average
        )

    # ========================================================
    # MAIN API USED BY APP.PY
    # ========================================================

    def identify_or_register(
        self,
        embedding,
        timestamp,
        track_id=None
    ):
        """
        Identify an existing person or register a new person.

        This method is intentionally kept compatible with app.py.

        Returns:

            {
                "face_id": ...,
                "is_new": ...,
                "similarity": ...,
                "valid": ...,
                "ready": ...,
                "samples": ...
            }
        """

        embedding = self.normalize_embedding(
            embedding
        )

        if embedding is None:

            return {
                "face_id": None,
                "is_new": False,
                "similarity": 0.0,
                "valid": False,
                "ready": False,
                "samples": 0,
            }

        # ----------------------------------------------------
        # If app supplies a track ID, use multi-frame samples.
        # ----------------------------------------------------

        if track_id is not None:

            track_id = int(track_id)

            # Already recognized?
            previous_result = (
                self.track_results.get(
                    track_id
                )
            )

            if previous_result is not None:

                return previous_result

            sample_count = self.add_sample(
                track_id,
                embedding
            )

            samples = self.get_samples(
                track_id
            )

            # ------------------------------------------------
            # First try matching the current embedding.
            #
            # This allows an already-known person to be
            # recognized without waiting unnecessarily for
            # registration.
            # ------------------------------------------------

            face_id, score = self.identify(
                embedding
            )

            if (
                face_id is not None
                and score >= self.similarity_threshold
            ):

                result = {
                    "face_id": face_id,
                    "is_new": False,
                    "similarity": float(score),
                    "valid": True,
                    "ready": True,
                    "samples": sample_count,
                    "existing": True,
                }

                self.track_results[
                    track_id
                ] = result

                return result

            # ------------------------------------------------
            # Not enough samples yet.
            # ------------------------------------------------

            if sample_count < self.samples_required:

                return {
                    "face_id": None,
                    "is_new": False,
                    "similarity": float(score),
                    "valid": True,
                    "ready": False,
                    "samples": sample_count,
                    "existing": False,
                }

            # ------------------------------------------------
            # Build average embedding from multiple frames.
            # ------------------------------------------------

            averaged = self.average_embedding(
                samples
            )

            if averaged is None:

                return {
                    "face_id": None,
                    "is_new": False,
                    "similarity": 0.0,
                    "valid": False,
                    "ready": False,
                    "samples": sample_count,
                }

            # ------------------------------------------------
            # Compare averaged embedding with database.
            # ------------------------------------------------

            face_id, average_score = self.identify(
                averaged
            )

            if (
                face_id is not None
                and average_score >= self.similarity_threshold
            ):

                result = {
                    "face_id": face_id,
                    "is_new": False,
                    "similarity": float(
                        average_score
                    ),
                    "valid": True,
                    "ready": True,
                    "samples": sample_count,
                    "existing": True,
                }

                self.track_results[
                    track_id
                ] = result

                return result

            # ------------------------------------------------
            # Unknown person.
            #
            # Register only after enough good samples.
            # ------------------------------------------------

            if average_score < self.registration_threshold:

                result = {
                    "face_id": None,
                    "is_new": False,
                    "similarity": float(
                        average_score
                    ),
                    "valid": True,
                    "ready": False,
                    "samples": sample_count,
                    "existing": False,
                }

                return result

            new_face_id = self.register(
                averaged,
                timestamp
            )

            if new_face_id is None:

                return {
                    "face_id": None,
                    "is_new": False,
                    "similarity": float(
                        average_score
                    ),
                    "valid": False,
                    "ready": False,
                    "samples": sample_count,
                }

            result = {
                "face_id": new_face_id,
                "is_new": True,
                "similarity": float(
                    average_score
                ),
                "valid": True,
                "ready": True,
                "samples": sample_count,
                "existing": False,
            }

            self.track_results[
                track_id
            ] = result

            return result

        # ====================================================
        # FALLBACK: no track ID
        # ====================================================

        face_id, score = self.identify(
            embedding
        )

        if face_id is not None:

            return {
                "face_id": face_id,
                "is_new": False,
                "similarity": float(score),
                "valid": True,
                "ready": True,
                "samples": 1,
                "existing": True,
            }

        # Without track-specific samples, register directly
        # only if the configured registration threshold permits.
        if score < self.registration_threshold:

            return {
                "face_id": None,
                "is_new": False,
                "similarity": float(score),
                "valid": True,
                "ready": False,
                "samples": 1,
                "existing": False,
            }

        face_id = self.register(
            embedding,
            timestamp
        )

        if face_id is None:

            return {
                "face_id": None,
                "is_new": False,
                "similarity": float(score),
                "valid": False,
                "ready": False,
                "samples": 1,
            }

        return {
            "face_id": face_id,
            "is_new": True,
            "similarity": float(score),
            "valid": True,
            "ready": True,
            "samples": 1,
            "existing": False,
        }