class OccupancyTracker:
    """Maintain current occupancy and cumulative entry/exit counts."""

    def __init__(self):
        self.current_occupancy = 0
        self.total_entries = 0
        self.total_exits = 0

    def process_event(self, event_type):
        if event_type == "ENTRY":
            self.total_entries += 1
            self.current_occupancy += 1

        elif event_type == "EXIT":
            self.total_exits += 1
            self.current_occupancy = max(
                0,
                self.current_occupancy - 1,
            )

    def get_summary(self):
        return {
            "current_occupancy": self.current_occupancy,
            "total_entries": self.total_entries,
            "total_exits": self.total_exits,
        }
