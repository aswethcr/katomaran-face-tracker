from src.occupancy import OccupancyTracker


tracker = OccupancyTracker()


events = [
    "ENTRY",
    "ENTRY",
    "ENTRY",
    "EXIT",
    "ENTRY",
    "EXIT"
]


for event in events:

    tracker.process_event(event)

    print(
        event,
        "→ occupancy:",
        tracker.occupancy
    )


print()
print("Final summary:")
print(tracker.get_summary())