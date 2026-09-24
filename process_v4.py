import csv
import json
from pathlib import Path

from inference import InferencePipeline


# ============================================================
# CONFIGURATION
# ============================================================

WORKSPACE = "venu-sudha"

WORKFLOW_ID = (
    "badminton-singles-players-"
    "vbadminton-singles-players-2-yolo26n-t1-logic"
)

INPUT_VIDEO = Path("video/badminton_test.mp4")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE = RESULTS_DIR / "badminton_coordinates_v4.csv"
JSON_FILE = RESULTS_DIR / "badminton_coordinates_v4.json"


# ============================================================
# STORAGE
# ============================================================

frame_counter = 0

all_frames = []
csv_rows = []


# ============================================================
# CALLBACK
# ============================================================

def on_prediction(prediction, video_frame):
    global frame_counter

    frame_counter += 1

    if frame_counter % 30 == 0:
        print(f"Processed frame: {frame_counter}")

    # --------------------------------------------------------
    # Court coordinates are directly available at the
    # top level of the workflow result.
    # --------------------------------------------------------

    coordinates = prediction.get("court_coordinates", [])

    frame_record = {
        "frame": frame_counter,
        "court_coordinates": []
    }

    for player in coordinates:

        clean_player = {
            "tracker_id": player.get("tracker_id"),
            "class_name": player.get("class_name"),
            "court_x_m": player.get("court_x_m"),
            "court_y_m": player.get("court_y_m"),
        }

        frame_record["court_coordinates"].append(clean_player)

        csv_rows.append({
            "frame": frame_counter,
            "tracker_id": player.get("tracker_id"),
            "class_name": player.get("class_name"),
            "court_x_m": player.get("court_x_m"),
            "court_y_m": player.get("court_y_m"),
        })

    all_frames.append(frame_record)


# ============================================================
# CHECK INPUT
# ============================================================

if not INPUT_VIDEO.exists():
    raise FileNotFoundError(
        f"Input video not found:\n{INPUT_VIDEO.resolve()}"
    )


# ============================================================
# START
# ============================================================

print("=" * 70)
print("BADMINTON VERSION 4 FULL VIDEO PROCESSING")
print("=" * 70)

print(f"Workflow : {WORKFLOW_ID}")
print(f"Video    : {INPUT_VIDEO}")
print(f"CSV      : {CSV_FILE}")
print(f"JSON     : {JSON_FILE}")
print()


pipeline = InferencePipeline.init_with_workflow(
    video_reference=str(INPUT_VIDEO),
    workspace_name=WORKSPACE,
    workflow_id=WORKFLOW_ID,
    on_prediction=on_prediction,
)


print("Starting workflow...")
pipeline.start()

print("Processing video...")
pipeline.join()


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("PROCESSING COMPLETE")
print("=" * 70)

frames_with_coordinates = len(
    [
        frame
        for frame in all_frames
        if len(frame["court_coordinates"]) > 0
    ]
)


# ============================================================
# SAVE JSON
# ============================================================

with open(JSON_FILE, "w", encoding="utf-8") as f:
    json.dump(all_frames, f, indent=2)


# ============================================================
# SAVE CSV
# ============================================================

with open(
    CSV_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "frame",
            "tracker_id",
            "class_name",
            "court_x_m",
            "court_y_m",
        ],
    )

    writer.writeheader()
    writer.writerows(csv_rows)


# ============================================================
# FINAL OUTPUT
# ============================================================

print(f"Frames processed       : {frame_counter}")
print(f"Frames with coordinates: {frames_with_coordinates}")
print(f"Coordinate rows        : {len(csv_rows)}")

print()
print(f"JSON saved : {JSON_FILE.resolve()}")
print(f"CSV saved  : {CSV_FILE.resolve()}")

print("=" * 70)