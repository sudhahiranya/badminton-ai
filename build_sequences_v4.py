import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

INPUT_CSV = Path(
    "results/badminton_movement_features_v4_clean.csv"
)

OUTPUT_NPZ = Path(
    "results/movement_sequences_v4.npz"
)

OUTPUT_META = Path(
    "results/movement_sequences_v4_metadata.csv"
)

COURT_WIDTH_M = 5.18
COURT_LENGTH_M = 13.40

FPS = 30.0

# 60 frames = 2 seconds
SEQUENCE_LENGTH = 60

# 30-frame stride = 1 second
STRIDE = 30


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_CSV)

required = {
    "segment_id",
    "frame",
    "near_x_m",
    "near_y_m",
    "far_x_m",
    "far_y_m",
    "both_detected",
}

missing = required - set(df.columns)

if missing:
    raise ValueError(
        f"Missing columns: {sorted(missing)}"
    )


# ============================================================
# COURT BOUNDS
#
# Small calibration overshoots are projected back into
# the physical singles court.
# ============================================================

for col in ["near_x_m", "far_x_m"]:
    df[col] = df[col].clip(
        0.0,
        COURT_WIDTH_M
    )

for col in ["near_y_m", "far_y_m"]:
    df[col] = df[col].clip(
        0.0,
        COURT_LENGTH_M
    )


# ============================================================
# ONLY USE FRAMES WHERE BOTH PLAYERS ARE PRESENT
# ============================================================

df = df[
    df["both_detected"].astype(bool)
].copy()

df = df.sort_values(
    ["segment_id", "frame"]
)


# ============================================================
# FEATURE CREATION
# ============================================================

FEATURE_NAMES = [
    "near_x_norm",
    "near_y_norm",
    "far_x_norm",
    "far_y_norm",
    "near_vx_mps",
    "near_vy_mps",
    "far_vx_mps",
    "far_vy_mps",
    "near_speed_mps",
    "far_speed_mps",
    "player_distance_m",
]


# ============================================================
# FIND CONTIGUOUS RUNS
# ============================================================

def contiguous_runs(segment_df):

    frames = segment_df["frame"].to_numpy()

    if len(frames) == 0:
        return []

    breaks = np.where(
        np.diff(frames) != 1
    )[0]

    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(frames) - 1]

    return [
        segment_df.iloc[start:end + 1].copy()
        for start, end in zip(starts, ends)
    ]


# ============================================================
# BUILD SEQUENCES
# ============================================================

sequences = []
metadata = []

sequence_id = 0


for segment_id in sorted(
    df["segment_id"].unique()
):

    segment = df[
        df["segment_id"] == segment_id
    ].copy()

    runs = contiguous_runs(segment)

    for run in runs:

        if len(run) < SEQUENCE_LENGTH:
            continue

        run = run.sort_values("frame").reset_index(
            drop=True
        )

        # -----------------------------------------------
        # Recalculate motion from positions.
        # This avoids depending on the previous capped
        # speed values.
        # -----------------------------------------------

        near_x = run["near_x_m"].to_numpy(float)
        near_y = run["near_y_m"].to_numpy(float)

        far_x = run["far_x_m"].to_numpy(float)
        far_y = run["far_y_m"].to_numpy(float)

        near_vx = np.gradient(
            near_x,
            1.0 / FPS
        )

        near_vy = np.gradient(
            near_y,
            1.0 / FPS
        )

        far_vx = np.gradient(
            far_x,
            1.0 / FPS
        )

        far_vy = np.gradient(
            far_y,
            1.0 / FPS
        )

        near_speed = np.sqrt(
            near_vx ** 2 +
            near_vy ** 2
        )

        far_speed = np.sqrt(
            far_vx ** 2 +
            far_vy ** 2
        )

        player_distance = np.sqrt(
            (near_x - far_x) ** 2 +
            (near_y - far_y) ** 2
        )

        # Normalize coordinates to [0,1]
        near_x_norm = (
            near_x / COURT_WIDTH_M
        )

        near_y_norm = (
            near_y / COURT_LENGTH_M
        )

        far_x_norm = (
            far_x / COURT_WIDTH_M
        )

        far_y_norm = (
            far_y / COURT_LENGTH_M
        )

        feature_matrix = np.column_stack([
            near_x_norm,
            near_y_norm,
            far_x_norm,
            far_y_norm,
            near_vx,
            near_vy,
            far_vx,
            far_vy,
            near_speed,
            far_speed,
            player_distance,
        ])

        # -----------------------------------------------
        # Replace any numerical infinities/nans.
        # -----------------------------------------------

        feature_matrix = np.nan_to_num(
            feature_matrix,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # -----------------------------------------------
        # Sliding windows
        # -----------------------------------------------

        for start in range(
            0,
            len(run) - SEQUENCE_LENGTH + 1,
            STRIDE,
        ):

            end = start + SEQUENCE_LENGTH

            sequence = feature_matrix[
                start:end
            ]

            sequences.append(sequence)

            metadata.append({
                "sequence_id": sequence_id,
                "segment_id": segment_id,
                "start_frame": int(
                    run.iloc[start]["frame"]
                ),
                "end_frame": int(
                    run.iloc[end - 1]["frame"]
                ),
                "duration_seconds": (
                    SEQUENCE_LENGTH / FPS
                ),
            })

            sequence_id += 1


# ============================================================
# CREATE ARRAY
# ============================================================

if not sequences:
    raise RuntimeError(
        "No 60-frame sequences were found."
    )

X = np.stack(
    sequences
).astype(np.float32)

metadata_df = pd.DataFrame(
    metadata
)


# ============================================================
# SAVE
# ============================================================

np.savez_compressed(
    OUTPUT_NPZ,
    X=X,
    feature_names=np.array(
        FEATURE_NAMES
    ),
)

metadata_df.to_csv(
    OUTPUT_META,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print("=" * 70)
print("BADMINTON MOVEMENT SEQUENCES — VERSION 4")
print("=" * 70)

print(f"Sequences          : {X.shape[0]}")
print(f"Frames / sequence  : {X.shape[1]}")
print(f"Features / frame   : {X.shape[2]}")

print(
    f"Sequence duration  : "
    f"{SEQUENCE_LENGTH / FPS:.2f} seconds"
)

print(
    f"Stride             : "
    f"{STRIDE / FPS:.2f} seconds"
)

print()
print("Feature order:")

for i, name in enumerate(
    FEATURE_NAMES
):
    print(
        f"{i:2d}: {name}"
    )

print()
print(
    "Array shape:",
    X.shape
)

print()
print(
    f"Saved sequences : {OUTPUT_NPZ.resolve()}"
)

print(
    f"Saved metadata  : {OUTPUT_META.resolve()}"
)

print("=" * 70)