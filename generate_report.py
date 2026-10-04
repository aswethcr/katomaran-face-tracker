import sqlite3
from collections import Counter


DB_PATH = "database/faces.db"


# --------------------------------------------------
# Connect to database
# --------------------------------------------------

conn = sqlite3.connect(DB_PATH)

cursor = conn.cursor()


# --------------------------------------------------
# Total unique faces
# --------------------------------------------------

cursor.execute("""
    SELECT COUNT(*)
    FROM faces
""")

total_faces = cursor.fetchone()[0]


# --------------------------------------------------
# All events
# --------------------------------------------------

cursor.execute("""
    SELECT face_id, event_type, timestamp
    FROM events
    ORDER BY timestamp
""")

events = cursor.fetchall()


# --------------------------------------------------
# Count event types
# --------------------------------------------------

event_counts = Counter(
    event[1]
    for event in events
)


total_entries = event_counts["ENTRY"]
total_exits = event_counts["EXIT"]


# --------------------------------------------------
# Calculate occupancy
# --------------------------------------------------

occupancy = 0

for _, event_type, _ in events:

    if event_type == "ENTRY":
        occupancy += 1

    elif event_type == "EXIT":
        occupancy = max(
            0,
            occupancy - 1
        )


# --------------------------------------------------
# Print report
# --------------------------------------------------

print()
print("======================================")
print("       KATOMARAN FACE TRACKER")
print("             FINAL REPORT")
print("======================================")

print(
    f"Unique faces detected : {total_faces}"
)

print(
    f"Total entries         : {total_entries}"
)

print(
    f"Total exits           : {total_exits}"
)

print(
    f"Current occupancy     : {occupancy}"
)

print("--------------------------------------")

print("EVENTS")
print("--------------------------------------")


for event_id, (face_id, event_type, timestamp) in enumerate(
    events,
    start=1
):

    print(
        f"{event_id:03d} | "
        f"{face_id:<12} | "
        f"{event_type:<5} | "
        f"{timestamp}"
    )


print("======================================")


conn.close()