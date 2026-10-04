from datetime import datetime
import numpy as np


class FaceRegistrar:

    def __init__(self, database, similarity_threshold=0.45):
        self.database = database
        self.similarity_threshold = similarity_threshold

    @staticmethod
    def cosine_similarity(a, b):

        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)

        if a_norm == 0 or b_norm == 0:
            return 0.0

        return float(
            np.dot(a, b) / (a_norm * b_norm)
        )

    def find_match(self, embedding):

        stored_faces = self.database.get_all_faces()

        best_face_id = None
        best_similarity = 0.0

        for face in stored_faces:

            similarity = self.cosine_similarity(
                embedding,
                face["embedding"]
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_face_id = face["face_id"]

        if (
            best_face_id is not None
            and best_similarity >= self.similarity_threshold
        ):
            return best_face_id, best_similarity

        return None, best_similarity

    def register_or_identify(self, embedding):

        face_id, similarity = self.find_match(embedding)

        if face_id is not None:
            return face_id, False, similarity

        existing_faces = self.database.get_all_faces()

        next_number = len(existing_faces) + 1

        new_face_id = f"FACE_{next_number:04d}"

        timestamp = datetime.now().isoformat()

        self.database.add_face(
            new_face_id,
            embedding,
            timestamp
        )

        return new_face_id, True, 1.0