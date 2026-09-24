import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_CSV = Path(
    "results/badminton_movement_features_v4_clean.csv"
)

OUTPUT_NPZ = Path(
    "results/movement_sequences_v4_interpolated.npz"
)

OUTPUT_META = Path(
    "results/movement_sequences_v4_interpolated_metadata.csv"
)

FPS = 30.0

COURT_WIDTH_M = 5.18
COURT_LENGTH_M = 13.40

SEQUENCE_LENGTH = 60       # 2 seconds
STRIDE = 15                # 0.5 seconds

MAX_INTERPOLATED_FRAMES = 6
MAX_INTERPOLATION_GAP = 3


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_CSV)

required = {
    "segment_id",
    "frame",
    "near_x_m",
    "near_y_m",
    "far_x_m",
    "far_y_m",
}

missing = required - set(df.columns)

if missing:
    raise ValueError(
        f"Missing columns: {sorted(missing)}"
    )

df = df.sort_values(
    ["segment_id", "frame"]
).copy()


# ============================================================
# CLIP SMALL CALIBRATION OVERSHOOTS
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
# INTERPOLATE ONLY SHORT INTERNAL GAPS
# ============================================================

position_columns = [
    "near_x_m",
    "near_y_m",
    "far_x_m",
    "far_y_m",
]

for col in position_columns:

    original = df[col].copy()

    df[col] = (
        df.groupby("segment_id")[col]
        .transform(
            lambda s: s.interpolate(
                method="linear",
                limit=MAX_INTERPOLATION_GAP,
                limit_area="inside"
            )
        )
    )

    # Mark values that were filled by interpolation.
    df[f"{col}_imputed"] = (
        original.isna()
        & df[col].notna()
    )


# ============================================================
# CREATE PLAYER-LEVEL IMPUTATION FLAGS
# ============================================================

df["near_imputed"] = (
    df["near_x_m_imputed"]
    | df["near_y_m_imputed"]
)

df["far_imputed"] = (
    df["far_x_m_imputed"]
    | df["far_y_m_imputed"]
)

df["total_imputed"] = (
    df["near_imputed"].astype(int)
    + df["far_imputed"].astype(int)
)


# ============================================================
# FIND COMPLETE RUNS
# ============================================================

position_complete = (
    df[
        [
            "near_x_m",
            "near_y_m",
            "far_x_m",
            "far_y_m",
        ]
    ]
    .notna()
    .all(axis=1)
)

df["positions_complete"] = position_complete


# ============================================================
# FEATURES
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


sequences = []
metadata = []

sequence_id = 0


# ============================================================
# PROCESS EACH TRACKING SEGMENT
# ============================================================

for segment_id in sorted(
    df["segment_id"].unique()
):

    segment = df[
        df["segment_id"] == segment_id
    ].copy()

    segment = segment.sort_values(
        "frame"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Find runs where all four positions are available.
    # --------------------------------------------------------

    valid = segment[
        "positions_complete"
    ].to_numpy()

    if len(valid) == 0:
        continue

    breaks = np.where(
        np.diff(valid.astype(int)) != 0
    )[0]

    run_starts = np.r_[0, breaks + 1]
    run_ends = np.r_[breaks, len(segment) - 1]

    for start, end in zip(
        run_starts,
        run_ends
    ):

        run = segment.iloc[
            start:end + 1
        ].copy()

        # Only use runs where positions are complete.
        if not bool(
            run["positions_complete"].all()
        ):
            continue

        # ----------------------------------------------------
        # Require consecutive video frames.
        # ----------------------------------------------------

        frames = run[
            "frame"
        ].to_numpy()

        if len(frames) < SEQUENCE_LENGTH:
            continue

        # Split if a frame is missing from the timeline.
        frame_breaks = np.where(
            np.diff(frames) != 1
        )[0]

        sub_starts = np.r_[0, frame_breaks + 1]
        sub_ends = np.r_[
            frame_breaks,
            len(run) - 1
        ]

        for sub_start, sub_end in zip(
            sub_starts,
            sub_ends
        ):

            subrun = run.iloc[
                sub_start:sub_end + 1
            ].copy()

            if len(subrun) < SEQUENCE_LENGTH:
                continue

            # ------------------------------------------------
            # Convert positions to arrays
            # ------------------------------------------------

            near_x = subrun[
                "near_x_m"
            ].to_numpy(float)

            near_y = subrun[
                "near_y_m"
            ].to_numpy(float)

            far_x = subrun[
                "far_x_m"
            ].to_numpy(float)

            far_y = subrun[
                "far_y_m"
            ].to_numpy(float)

            near_imputed = subrun[
                "near_imputed"
            ].to_numpy(bool)

            far_imputed = subrun[
                "far_imputed"
            ].to_numpy(bool)

            # ------------------------------------------------
            # Sliding windows
            # ------------------------------------------------

            for window_start in range(
                0,
                len(subrun) - SEQUENCE_LENGTH + 1,
                STRIDE
            ):

                window_end = (
                    window_start +
                    SEQUENCE_LENGTH
                )

                wx = near_x[
                    window_start:window_end
                ]

                wy = near_y[
                    window_start:window_end
                ]

                fx = far_x[
                    window_start:window_end
                ]

                fy = far_y[
                    window_start:window_end
                ]

                wi_near = near_imputed[
                    window_start:window_end
                ]

                wi_far = far_imputed[
                    window_start:window_end
                ]

                total_interpolated = int(
                    wi_near.sum() +
                    wi_far.sum()
                )

                # --------------------------------------------
                # Reject sequences with too much interpolation
                # --------------------------------------------

                if (
                    total_interpolated
                    > MAX_INTERPOLATED_FRAMES
                ):
                    continue

                # --------------------------------------------
                # Velocity
                # --------------------------------------------

                near_vx = np.gradient(
                    wx,
                    1.0 / FPS
                )

                near_vy = np.gradient(
                    wy,
                    1.0 / FPS
                )

                far_vx = np.gradient(
                    fx,
                    1.0 / FPS
                )

                far_vy = np.gradient(
                    fy,
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
                    (wx - fx) ** 2 +
                    (wy - fy) ** 2
                )

                # --------------------------------------------
                # Normalize court positions
                # --------------------------------------------

                near_x_norm = (
                    wx /
                    COURT_WIDTH_M
                )

                near_y_norm = (
                    wy /
                    COURT_LENGTH_M
                )

                far_x_norm = (
                    fx /
                    COURT_WIDTH_M
                )

                far_y_norm = (
                    fy /
                    COURT_LENGTH_M
                )

                # --------------------------------------------
                # Stack features
                # --------------------------------------------

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

                feature_matrix = np.nan_to_num(
                    feature_matrix,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0
                )

                sequences.append(
                    feature_matrix.astype(
                        np.float32
                    )
                )

                metadata.append({
                    "sequence_id": sequence_id,
                    "segment_id": int(
                        segment_id
                    ),
                    "start_frame": int(
                        subrun.iloc[
                            window_start
                        ]["frame"]
                    ),
                    "end_frame": int(
                        subrun.iloc[
                            window_end - 1
                        ]["frame"]
                    ),
                    "duration_seconds": (
                        SEQUENCE_LENGTH /
                        FPS
                    ),
                    "interpolated_frames": (
                        total_interpolated
                    ),
                })

                sequence_id += 1


# ============================================================
# CHECK
# ============================================================

if not sequences:
    raise RuntimeError(
        "No valid sequences were created."
    )


# ============================================================
# SAVE
# ============================================================

X = np.stack(
    sequences
).astype(np.float32)

metadata_df = pd.DataFrame(
    metadata
)

np.savez_compressed(
    OUTPUT_NPZ,
    X=X,
    feature_names=np.array(
        FEATURE_NAMES
    ),
)

metadata_df.to_csv(
    OUTPUT_META,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("=" * 70)
print("BADMINTON MOVEMENT SEQUENCES — V4 INTERPOLATED")
print("=" * 70)

print(
    f"Sequences          : {X.shape[0]}"
)

print(
    f"Frames / sequence  : {X.shape[1]}"
)

print(
    f"Features / frame   : {X.shape[2]}"
)

print(
    f"Sequence duration  : "
    f"{SEQUENCE_LENGTH / FPS:.2f} sec"
)

print(
    f"Stride             : "
    f"{STRIDE / FPS:.2f} sec"
)

print(
    f"Max allowed imputed frames: "
    f"{MAX_INTERPOLATED_FRAMES}"
)

print()

print(
    "Sequences by segment:"
)

print(
    metadata_df[
        "segment_id"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)

print()

print(
    "Interpolation summary:"
)

print(
    metadata_df[
        "interpolated_frames"
    ].describe().round(2).to_string()
)

print()
print(
    "Array shape:",
    X.shape
)

print()
print(
    f"Saved sequences: "
    f"{OUTPUT_NPZ.resolve()}"
)

print(
    f"Saved metadata : "
    f"{OUTPUT_META.resolve()}"
)

print("=" * 70)