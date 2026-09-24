import pandas as pd
import numpy as np

CSV = "results/badminton_movement_features_v4_clean.csv"

df = pd.read_csv(CSV)

print("=" * 80)
print("SEQUENCE GAP ANALYSIS")
print("=" * 80)

for player in ["near", "far"]:

    detected_col = f"{player}_detected"

    print()
    print("=" * 80)
    print(player.upper())
    print("=" * 80)

    all_gaps = []

    for segment_id in sorted(df["segment_id"].unique()):

        segment = df[
            df["segment_id"] == segment_id
        ].sort_values("frame").copy()

        detected = segment[detected_col].astype(bool).to_numpy()

        gap_lengths = []
        current_gap = 0

        for value in detected:

            if value:
                if current_gap > 0:
                    gap_lengths.append(current_gap)
                    current_gap = 0
            else:
                current_gap += 1

        if current_gap > 0:
            gap_lengths.append(current_gap)

        all_gaps.extend(gap_lengths)

        print(
            f"Segment {segment_id}: "
            f"max missing gap = "
            f"{max(gap_lengths, default=0)} frames"
        )

    if all_gaps:

        s = pd.Series(all_gaps)

        print()
        print("Missing-gap statistics:")
        print(
            s.describe(
                percentiles=[
                    .50,
                    .75,
                    .90,
                    .95,
                    .99
                ]
            ).round(2).to_string()
        )

        print()
        print("Gap counts:")

        for threshold in [1, 2, 3, 5, 10, 15, 30]:

            count = int(
                (s <= threshold).sum()
            )

            print(
                f"<= {threshold:2d} frames: "
                f"{count}"
            )

print()
print("=" * 80)