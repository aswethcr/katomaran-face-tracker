import numpy as np
from insightface.app import FaceAnalysis


class FaceRecognizer:

    def __init__(self):
        self.app = FaceAnalysis(
            name="buffalo_l"
        )

        self.app.prepare(
            ctx_id=0,
            det_size=(640, 640)
        )

    def get_faces(self, frame):
        """
        Detect faces and return InsightFace objects.
        """

        return self.app.get(frame)

    @staticmethod
    def get_embedding(face):
        """
        Return normalized face embedding.
        """

        embedding = face.embedding.astype(
            np.float32
        )

        norm = np.linalg.norm(embedding)

        if norm == 0:
            return embedding

        return embedding / norm