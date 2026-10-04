def crop_face(frame, bbox):

    x1, y1, x2, y2 = bbox

    height, width = frame.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2]