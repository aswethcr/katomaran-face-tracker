import cv2


class VideoReader:
    def __init__(self, source):
        self.source = source
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()

    def get_fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS)

    def get_width(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    def get_height(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))