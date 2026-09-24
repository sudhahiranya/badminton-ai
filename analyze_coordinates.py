import pandas as pd
import os

CSV_FILE = "results/badminton_coordinates.csv"

print("=" * 70)
print("BADMINTON COORDINATE DATA ANALYSIS")
print("=" * 70)

if not os.path.exists(CSV_FILE):
    print(f"\nERROR: File not found:")
    print(CSV_FILE)
    raise SystemExit

df = pd.read_csv(CSV_FILE)

print("\n1. DATASET SIZE")
print("-" * 70)
print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")

print("\n2. COLUMNS")
print("-" * 70)

for column in df.columns:
    print(f"- {column}")

print("\n3. FIRST 10 ROWS")
print("-" * 70)
print(df.head(10).to_string(index=False))

print("\n4. TRACKER IDs")
print("-" * 70)

far_ids = df["far_tracker_id"].dropna().unique()
near_ids = df["near_tracker_id"].dropna().unique()

print("Far tracker IDs:")
print(far_ids)

print("\nNear tracker IDs:")
print(near_ids)

print("\n5. NON-MISSING COORDINATES")
print("-" * 70)

far_valid = df[
    df["far_court_x_m"].notna()
    & df["far_court_y_m"].notna()
]

near_valid = df[
    df["near_court_x_m"].notna()
    & df["near_court_y_m"].notna()
]

print(f"Far coordinate rows  : {len(far_valid)}")
print(f"Near coordinate rows : {len(near_valid)}")

print("\n6. COURT X RANGE")
print("-" * 70)

if len(far_valid) > 0:
    print(
        "Far X:",
        far_valid["far_court_x_m"].min(),
        "to",
        far_valid["far_court_x_m"].max()
    )

if len(near_valid) > 0:
    print(
        "Near X:",
        near_valid["near_court_x_m"].min(),
        "to",
        near_valid["near_court_x_m"].max()
    )

print("\n7. COURT Y RANGE")
print("-" * 70)

if len(far_valid) > 0:
    print(
        "Far Y:",
        far_valid["far_court_y_m"].min(),
        "to",
        far_valid["far_court_y_m"].max()
    )

if len(near_valid) > 0:
    print(
        "Near Y:",
        near_valid["near_court_y_m"].min(),
        "to",
        near_valid["near_court_y_m"].max()
    )

print("\n8. MISSING VALUES")
print("-" * 70)

print(df.isna().sum())

print("\n9. ROWS WITH BOTH PLAYERS")
print("-" * 70)

both_players = df[
    df["far_court_x_m"].notna()
    & df["far_court_y_m"].notna()
    & df["near_court_x_m"].notna()
    & df["near_court_y_m"].notna()
]

print(f"Frames with both players: {len(both_players)}")

print("\n10. ROWS WITH AT LEAST ONE PLAYER")
print("-" * 70)

at_least_one = df[
    df["far_court_x_m"].notna()
    | df["far_court_y_m"].notna()
    | df["near_court_x_m"].notna()
    | df["near_court_y_m"].notna()
]

print(f"Frames with at least one player: {len(at_least_one)}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)