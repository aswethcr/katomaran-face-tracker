from src.event_manager import EventManager


manager = EventManager(
    line_y=500
)


# Simulate a person moving downward
positions = [
    400,
    420,
    450,
    480,
    510,
    530
]


for frame, y in enumerate(
    positions,
    start=1
):

    event = manager.update(
        track_id=1,
        face_id="FACE_0001",
        center_y=y,
        frame_number=frame
    )

    if event:
        print(event)