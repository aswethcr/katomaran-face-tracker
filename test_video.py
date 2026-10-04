import cv2

VIDEO_PATH = "data/videos/record_20250620_183903.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Could not open video")

print("Video opened successfully!")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("FPS:", fps)
print("Resolution:", width, "x", height)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Katomaran Video Test", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()