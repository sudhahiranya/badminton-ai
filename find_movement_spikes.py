import pandas as pd
import numpy as np

INPUT_CSV = "results/badminton_movement_features_v4.csv"

df = pd.read_csv(INPUT_CSV)

all_spikes = []

for player in ["near", "far"]:

    x_col = f"{player}_x_m"
    y_col = f"{player}_y_m"

    temp = df[
        ["segment_id", "frame", x_col, y_col]
    ].dropna().copy()

    temp = temp.sort_values(
        ["segment_id", "frame"]
    )

    # Previous frame within the same segment
    temp["previous_frame"] = (
        temp.groupby("segment_id")["frame"]
        .shift(1)
    )

    temp["dx"] = (
        temp.groupby("segment_id")[x_col]
        .diff()
    )

    temp["dy"] = (
        temp.groupby("segment_id")[y_col]
        .diff()
    )

    # Only compare consecutive frames
    temp = temp[
        temp["frame"] == temp["previous_frame"] + 1
    ].copy()

    temp["step_m"] = np.sqrt(
        temp["dx"] ** 2 +
        temp["dy"] ** 2
    )

    temp["speed_mps"] = temp["step_m"] * 30.0

    temp["player"] = player

    all_spikes.append(
        temp[
            [
                "player",
                "segment_id",
                "frame",
                "step_m",
                "speed_mps",
                x_col,
                y_col,
            ]
        ]
    )

result = pd.concat(
    all_spikes,
    ignore_index=True
)

result = result.sort_values(
    "step_m",
    ascending=False
)

print("=" * 90)
print("LARGEST FRAME-TO-FRAME MOVEMENT SPIKES")
print("=" * 90)

print(
    result.head(30).to_string(
        index=False
    )
)

print()
print("=" * 90)
print("SPEED SUMMARY")
print("=" * 90)

print(
    result.groupby("player")["speed_mps"]
    .describe(
        percentiles=[0.50, 0.90, 0.95, 0.99]
    )
    .round(3)
    .to_string()
)