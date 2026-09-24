import cv2
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

INPUT_CSV = Path("results/badminton_coordinates_v4.csv")
INPUT_VIDEO = Path("video/badminton_test.mp4")

OUTPUT_CSV = Path(
    "results/badminton_movement_features_v4_clean.csv"
)

FPS = 30.0

COURT_MID_Y = 6.70

# A player cannot realistically jump several meters
# between consecutive 30 FPS frames.
MAX_STEP_M = 0.50


# Continuous video segments discovered earlier.
SEGMENTS = [
    (1, 172, 680),
    (2, 721, 1020),
    (3, 951, 1152),
    (4, 1263, 1429),
    (5, 1470, 1802),
]


# ============================================================
# LOAD RAW COORDINATES
# ============================================================

df = pd.read_csv(INPUT_CSV)

required = {
    "frame",
    "tracker_id",
    "class_name",
    "court_x_m",
    "court_y_m",
}

missing = required - set(df.columns)

if missing:
    raise ValueError(
        f"Missing columns: {sorted(missing)}"
    )

df["tracker_id"] = df["tracker_id"].astype(int)


# ============================================================
# ASSIGN SEGMENT
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
# ASSIGN NEAR/FAR BY COURT SIDE
# ============================================================

def side_from_y(y):
    if y < COURT_MID_Y:
        return "near"
    else:
        return "far"


df["side"] = df["court_y_m"].apply(side_from_y)


# ============================================================
# REMOVE DUPLICATE SAME-SIDE DETECTIONS
#
# If two detections are assigned to the same side in one
# frame, keep the one with the stronger detector confidence
# if available. Otherwise keep the first.
# ============================================================

# Raw coordinate CSV currently does not contain confidence,
# so use the detection closest to the expected side center.

def select_one_per_side(group):

    output = []

    for side in ["near", "far"]:

        candidates = group[
            group["side"] == side
        ]

        if candidates.empty:
            continue

        # Expected side center
        expected_y = 3.35 if side == "near" else 10.05

        candidates = candidates.copy()

        candidates["side_distance"] = (
            candidates["court_y_m"] - expected_y
        ).abs()

        selected = candidates.sort_values(
            "side_distance"
        ).iloc[0]

        output.append(selected)

    if not output:
        return pd.DataFrame(
            columns=group.columns
        )

    return pd.DataFrame(output)


clean_rows = []

for frame, group in df.groupby(
    ["segment_id", "frame"],
    sort=True
):

    selected = select_one_per_side(group)

    for _, row in selected.iterrows():
        clean_rows.append(row)


clean = pd.DataFrame(clean_rows)


# ============================================================
# CREATE FRAME-LEVEL DATA
# ============================================================

rows = []

for segment_id, start_frame, end_frame in SEGMENTS:

    segment = clean[
        clean["segment_id"] == segment_id
    ]

    for frame in range(
        start_frame,
        end_frame + 1
    ):

        frame_data = segment[
            segment["frame"] == frame
        ]

        row = {
            "segment_id": segment_id,
            "frame": frame,
            "time_seconds": (frame - 1) / FPS,

            "near_x_m": np.nan,
            "near_y_m": np.nan,

            "far_x_m": np.nan,
            "far_y_m": np.nan,

            "near_tracker_id": np.nan,
            "far_tracker_id": np.nan,
        }

        near = frame_data[
            frame_data["side"] == "near"
        ]

        far = frame_data[
            frame_data["side"] == "far"
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
# REJECT IMPOSSIBLE ONE-FRAME JUMPS
# ============================================================

for player in ["near", "far"]:

    x = f"{player}_x_m"
    y = f"{player}_y_m"

    for segment_id in movement["segment_id"].unique():

        mask = (
            movement["segment_id"]
            == segment_id
        )

        segment = movement.loc[
            mask
        ].copy()

        dx = segment[x].diff()
        dy = segment[y].diff()

        step = np.sqrt(
            dx ** 2 +
            dy ** 2
        )

        bad = (
            step > MAX_STEP_M
        )

        bad_indices = segment.index[bad]

        # Invalidate only the endpoint of the
        # impossible jump.
        movement.loc[
            bad_indices,
            [x, y]
        ] = np.nan


# ============================================================
# DETECTION FLAGS
# ============================================================

movement["near_detected"] = (
    movement["near_x_m"].notna()
)

movement["far_detected"] = (
    movement["far_x_m"].notna()
)

movement["both_detected"] = (
    movement["near_detected"]
    & movement["far_detected"]
)


# ============================================================
# MOTION COLUMNS
# ============================================================

for player in ["near", "far"]:

    movement[
        f"{player}_speed_mps"
    ] = np.nan

    movement[
        f"{player}_acceleration_mps2"
    ] = np.nan

    movement[
        f"{player}_distance_m"
    ] = np.nan


# ============================================================
# CALCULATE MOTION
# ============================================================

for player in ["near", "far"]:

    x_col = f"{player}_x_m"
    y_col = f"{player}_y_m"

    speed_col = f"{player}_speed_mps"
    accel_col = f"{player}_acceleration_mps2"
    distance_col = f"{player}_distance_m"

    for segment_id in movement[
        "segment_id"
    ].unique():

        mask = (
            movement["segment_id"]
            == segment_id
        )

        indices = movement.loc[
            mask
        ].index

        cumulative_distance = 0.0

        previous_x = None
        previous_y = None
        previous_speed = None
        previous_frame = None

        for idx in indices:

            x = movement.at[idx, x_col]
            y = movement.at[idx, y_col]
            frame = int(
                movement.at[idx, "frame"]
            )

            # Missing position:
            # reset temporal motion state.
            if pd.isna(x) or pd.isna(y):

                movement.at[
                    idx,
                    distance_col
                ] = cumulative_distance

                previous_x = None
                previous_y = None
                previous_speed = None
                previous_frame = None

                continue

            # First valid position
            if previous_x is None:

                movement.at[
                    idx,
                    distance_col
                ] = cumulative_distance

                previous_x = float(x)
                previous_y = float(y)
                previous_frame = frame
                previous_speed = None

                continue

            frame_gap = (
                frame - previous_frame
            )

            if frame_gap != 1:

                movement.at[
                    idx,
                    distance_col
                ] = cumulative_distance

                previous_x = float(x)
                previous_y = float(y)
                previous_frame = frame
                previous_speed = None

                continue

            dt = 1.0 / FPS

            dx = float(x) - previous_x
            dy = float(y) - previous_y

            step = float(
                np.sqrt(
                    dx * dx +
                    dy * dy
                )
            )

            # Another safety check.
            if step > MAX_STEP_M:

                movement.at[
                    idx,
                    distance_col
                ] = cumulative_distance

                previous_x = float(x)
                previous_y = float(y)
                previous_frame = frame
                previous_speed = None

                continue

            speed = step / dt

            cumulative_distance += step

            movement.at[
                idx,
                speed_col
            ] = speed

            movement.at[
                idx,
                distance_col
            ] = cumulative_distance

            if previous_speed is not None:

                acceleration = (
                    speed -
                    previous_speed
                ) / dt

                # Prevent acceleration from being
                # calculated across an invalid jump.
                movement.at[
                    idx,
                    accel_col
                ] = acceleration

            previous_speed = speed

            previous_x = float(x)
            previous_y = float(y)
            previous_frame = frame


# ============================================================
# SAVE
# ============================================================

movement.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("=" * 70)
print("CLEAN MOVEMENT DATASET")
print("=" * 70)

print(
    f"Rows                    : {len(movement)}"
)

print(
    "Both players detected   : "
    f"{int(movement['both_detected'].sum())}"
)

print(
    "Near detected frames    : "
    f"{int(movement['near_detected'].sum())}"
)

print(
    "Far detected frames     : "
    f"{int(movement['far_detected'].sum())}"
)

print()
print("Maximum speeds:")

print(
    "Near:",
    round(
        movement["near_speed_mps"].max(),
        3
    ),
    "m/s"
)

print(
    "Far :",
    round(
        movement["far_speed_mps"].max(),
        3
    ),
    "m/s"
)

print()
print("Distance by segment:")

for segment_id in movement[
    "segment_id"
].unique():

    segment = movement[
        movement["segment_id"]
        == segment_id
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
print(
    f"Saved to: {OUTPUT_CSV.resolve()}"
)

print("=" * 70)