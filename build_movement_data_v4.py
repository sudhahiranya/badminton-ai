import cv2
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

INPUT_CSV = Path("results/badminton_coordinates_v4.csv")
INPUT_VIDEO = Path("video/badminton_test.mp4")

OUTPUT_CSV = Path("results/badminton_movement_features_v4.csv")


# Tracker IDs discovered from the V4 analysis
NEAR_IDS = {0, 3, 4, 6, 8}
FAR_IDS = {1, 2, 5, 7, 9}


# Continuous tracking segments
SEGMENTS = [
    (1, 172, 680),
    (2, 721, 1020),
    (3, 951, 1152),
    (4, 1263, 1429),
    (5, 1470, 1802),
]


# ============================================================
# LOAD VIDEO FPS
# ============================================================

cap = cv2.VideoCapture(str(INPUT_VIDEO))

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {INPUT_VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

cap.release()

if fps <= 0:
    fps = 30.0

print("=" * 70)
print("BUILDING BADMINTON MOVEMENT FEATURES — VERSION 4")
print("=" * 70)
print(f"Video FPS          : {fps:.3f}")
print(f"Video frames       : {total_video_frames}")


# ============================================================
# LOAD COORDINATES
# ============================================================

df = pd.read_csv(INPUT_CSV)

required_columns = {
    "frame",
    "tracker_id",
    "class_name",
    "court_x_m",
    "court_y_m",
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")


df["tracker_id"] = df["tracker_id"].astype(int)


# ============================================================
# ASSIGN LOGICAL PLAYER
# ============================================================

def logical_player(tracker_id):
    if tracker_id in NEAR_IDS:
        return "near"
    if tracker_id in FAR_IDS:
        return "far"
    return None


df["player"] = df["tracker_id"].apply(logical_player)

df = df[df["player"].notna()].copy()


# ============================================================
# ASSIGN TRACKING SEGMENT
# ============================================================

def get_segment(frame):
    for segment_id, start_frame, end_frame in SEGMENTS:
        if start_frame <= frame <= end_frame:
            return segment_id

    return np.nan


df["segment_id"] = df["frame"].apply(get_segment)

df = df[df["segment_id"].notna()].copy()
df["segment_id"] = df["segment_id"].astype(int)


# ============================================================
# CREATE FRAME-LEVEL TABLE
# ============================================================

rows = []

for segment_id, start_frame, end_frame in SEGMENTS:

    segment_frames = range(start_frame, end_frame + 1)

    segment_df = df[
        (df["segment_id"] == segment_id)
    ].copy()

    for frame in segment_frames:

        frame_df = segment_df[
            segment_df["frame"] == frame
        ]

        row = {
            "segment_id": segment_id,
            "frame": frame,
            "time_seconds": (frame - 1) / fps,

            "near_x_m": np.nan,
            "near_y_m": np.nan,

            "far_x_m": np.nan,
            "far_y_m": np.nan,

            "near_tracker_id": np.nan,
            "far_tracker_id": np.nan,
        }

        near = frame_df[
            frame_df["player"] == "near"
        ]

        far = frame_df[
            frame_df["player"] == "far"
        ]

        if not near.empty:

            n = near.iloc[0]

            row["near_x_m"] = n["court_x_m"]
            row["near_y_m"] = n["court_y_m"]
            row["near_tracker_id"] = n["tracker_id"]

        if not far.empty:

            f = far.iloc[0]

            row["far_x_m"] = f["court_x_m"]
            row["far_y_m"] = f["court_y_m"]
            row["far_tracker_id"] = f["tracker_id"]

        rows.append(row)


movement = pd.DataFrame(rows)


# ============================================================
# DETECTION FLAGS
# ============================================================

movement["near_detected"] = movement["near_x_m"].notna()
movement["far_detected"] = movement["far_x_m"].notna()

movement["both_detected"] = (
    movement["near_detected"]
    & movement["far_detected"]
)


# ============================================================
# INITIALIZE MOTION FEATURES
# ============================================================

motion_columns = [
    "near_speed_mps",
    "far_speed_mps",
    "near_acceleration_mps2",
    "far_acceleration_mps2",
    "near_distance_m",
    "far_distance_m",
]

for col in motion_columns:
    movement[col] = np.nan


# ============================================================
# CALCULATE MOTION WITHIN EACH CONTINUOUS SEGMENT
# ============================================================

for player in ["near", "far"]:

    x_col = f"{player}_x_m"
    y_col = f"{player}_y_m"

    speed_col = f"{player}_speed_mps"
    accel_col = f"{player}_acceleration_mps2"
    distance_col = f"{player}_distance_m"

    cumulative_distance = 0.0
    previous_speed = np.nan

    previous_x = None
    previous_y = None
    previous_frame = None

    for idx in movement.index:

        row = movement.loc[idx]

        x = row[x_col]
        y = row[y_col]
        frame = int(row["frame"])

        if pd.isna(x) or pd.isna(y):

            previous_x = None
            previous_y = None
            previous_frame = None
            previous_speed = np.nan

            movement.loc[idx, distance_col] = cumulative_distance

            continue

        # First valid detection
        if previous_x is None:

            movement.loc[idx, distance_col] = cumulative_distance

            previous_x = float(x)
            previous_y = float(y)
            previous_frame = frame
            previous_speed = np.nan

            continue

        frame_gap = frame - previous_frame

        # Only calculate instantaneous motion across
        # consecutive frames.
        if frame_gap == 1:

            dt = frame_gap / fps

            dx = float(x) - previous_x
            dy = float(y) - previous_y

            distance = float(np.sqrt(dx * dx + dy * dy))

            speed = distance / dt

            cumulative_distance += distance

            movement.loc[idx, speed_col] = speed
            movement.loc[idx, distance_col] = cumulative_distance

            if not np.isnan(previous_speed):

                acceleration = (
                    speed - previous_speed
                ) / dt

                movement.loc[idx, accel_col] = acceleration

            previous_speed = speed

        else:

            # Missing frames / cut / tracking reset
            # prevents us from inventing motion.
            movement.loc[idx, distance_col] = cumulative_distance

            previous_speed = np.nan

        previous_x = float(x)
        previous_y = float(y)
        previous_frame = frame


# ============================================================
# SAVE
# ============================================================

movement.to_csv(
    OUTPUT_CSV,
    index=False,
)

print()
print("=" * 70)
print("MOVEMENT DATASET CREATED")
print("=" * 70)

print(f"Rows                    : {len(movement)}")
print(f"Segments                : {movement['segment_id'].nunique()}")
print(
    f"Both players detected   : "
    f"{int(movement['both_detected'].sum())}"
)

print(
    f"Near detected frames    : "
    f"{int(movement['near_detected'].sum())}"
)

print(
    f"Far detected frames     : "
    f"{int(movement['far_detected'].sum())}"
)

print()
print("Total distance by segment/player:")

for segment_id in sorted(
    movement["segment_id"].unique()
):

    segment = movement[
        movement["segment_id"] == segment_id
    ]

    near_distance = (
        segment["near_distance_m"]
        .dropna()
        .max()
    )

    far_distance = (
        segment["far_distance_m"]
        .dropna()
        .max()
    )

    print(
        f"Segment {segment_id}: "
        f"near={near_distance:.2f} m, "
        f"far={far_distance:.2f} m"
    )

print()
print(f"Saved to: {OUTPUT_CSV.resolve()}")
print("=" * 70)