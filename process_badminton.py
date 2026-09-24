import os
import csv
import cv2
import numpy as np

from inference import InferencePipeline


# ============================================================
# BADMINTON AI - FULL VIDEO PROCESSING
# ============================================================

# -----------------------------
# ROBoflow settings
# -----------------------------

WORKSPACE = "venu-sudha"

# IMPORTANT:
# This is the CURRENT PUBLISHED workflow ID from your Roboflow URL.
WORKFLOW_ID = (
    "badminton-singles-players-vbadminton-singles-players-2-yolo26n-t1-logic"
)

# API key comes from PowerShell environment variable:
# $env:ROBOFLOW_API_KEY
API_KEY = os.environ.get("ROBOFLOW_API_KEY")


# -----------------------------
# Video settings
# -----------------------------

INPUT_VIDEO = "video/badminton_test.mp4"

OUTPUT_VIDEO = "results/badminton_annotated.mp4"

OUTPUT_CSV = "results/badminton_coordinates.csv"


# -----------------------------
# Debug settings
# -----------------------------

DEBUG_FRAMES = {1, 5, 100, 200}


# ============================================================
# CHECK BASIC SETTINGS
# ============================================================

if not API_KEY:
    raise RuntimeError(
        "ROBOFLOW_API_KEY is not set.\n\n"
        "Run this in PowerShell first:\n"
        "$env:ROBOFLOW_API_KEY='YOUR_API_KEY'"
    )


if not os.path.exists(INPUT_VIDEO):
    raise FileNotFoundError(
        f"Input video was not found:\n{INPUT_VIDEO}"
    )


# Create output directory
os.makedirs("results", exist_ok=True)


# ============================================================
# OPEN VIDEO INFORMATION
# ============================================================

cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video:\n{INPUT_VIDEO}"
    )


FPS = cap.get(cv2.CAP_PROP_FPS)
WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

cap.release()


print()
print("=" * 70)
print("BADMINTON AI VIDEO PROCESSING")
print("=" * 70)

print(f"Input video     : {INPUT_VIDEO}")
print(f"Resolution      : {WIDTH} x {HEIGHT}")
print(f"FPS             : {FPS}")
print(f"Total frames    : {TOTAL_FRAMES}")
print(f"Workspace       : {WORKSPACE}")
print(f"Workflow ID     : {WORKFLOW_ID}")
print(f"Output video    : {OUTPUT_VIDEO}")
print(f"Output CSV      : {OUTPUT_CSV}")

print("=" * 70)
print()


# ============================================================
# VIDEO WRITER
# ============================================================

video_writer = None


def initialize_video_writer(frame):
    """
    Create the output MP4 writer using the actual frame size.
    """

    global video_writer

    if video_writer is not None:
        return

    frame_height, frame_width = frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    video_writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        fourcc,
        FPS if FPS > 0 else 30.0,
        (frame_width, frame_height),
    )

    if not video_writer.isOpened():
        raise RuntimeError(
            f"Could not create output video:\n{OUTPUT_VIDEO}"
        )


# ============================================================
# CSV FILE
# ============================================================

csv_file = open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8",
)

csv_writer = csv.writer(csv_file)

csv_writer.writerow(
    [
        "frame",
        "time_seconds",

        "far_tracker_id",
        "far_court_x_m",
        "far_court_y_m",

        "near_tracker_id",
        "near_court_x_m",
        "near_court_y_m",
    ]
)


# ============================================================
# HELPER:
# Convert values to normal Python values
# ============================================================

def safe_float(value):
    """
    Safely convert a value to float.
    Returns None if conversion is impossible.
    """

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value):
    """
    Safely convert a value to int.
    """

    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ============================================================
# HELPER:
# Extract court-coordinate records
# ============================================================

def extract_court_coordinates(prediction):
    """
    Extract court coordinates from the Roboflow Workflow output.

    Expected Workflow output:

        court_coordinates

    Each player should contain information such as:

        tracker_id
        class
        court_x_m
        court_y_m

    The function handles several possible dictionary/list
    formats so the CSV generation does not depend on one
    exact serialization format.
    """

    if not isinstance(prediction, dict):
        return []

    data = prediction.get("court_coordinates")

    if data is None:
        return []

    records = []

    # --------------------------------------------------------
    # CASE 1:
    # court_coordinates is already a list
    # --------------------------------------------------------

    if isinstance(data, list):

        for item in data:

            if isinstance(item, dict):

                records.append(item)

            elif isinstance(item, (list, tuple)):

                # Try:
                # [tracker_id, class, x, y]

                if len(item) >= 4:

                    records.append(
                        {
                            "tracker_id": item[0],
                            "class": item[1],
                            "court_x_m": item[2],
                            "court_y_m": item[3],
                        }
                    )

    # --------------------------------------------------------
    # CASE 2:
    # court_coordinates is a dictionary
    # --------------------------------------------------------

    elif isinstance(data, dict):

        # Sometimes the actual records are stored under
        # a nested key.

        possible_keys = [
            "coordinates",
            "players",
            "detections",
            "results",
            "data",
        ]

        found_nested = False

        for key in possible_keys:

            nested = data.get(key)

            if isinstance(nested, list):

                found_nested = True

                for item in nested:

                    if isinstance(item, dict):
                        records.append(item)

                break

        # If there was no nested list, check whether the
        # dictionary itself represents one player.

        if not found_nested:

            if (
                "court_x_m" in data
                or "court_y_m" in data
                or "tracker_id" in data
            ):
                records.append(data)

            else:

                # Possible format:
                #
                # {
                #     "151": {
                #         "court_x_m": ...,
                #         "court_y_m": ...
                #     }
                # }

                for key, value in data.items():

                    if isinstance(value, dict):

                        item = dict(value)

                        if "tracker_id" not in item:
                            item["tracker_id"] = key

                        records.append(item)

    # --------------------------------------------------------
    # Normalize field names
    # --------------------------------------------------------

    normalized = []

    for item in records:

        if not isinstance(item, dict):
            continue

        tracker_id = (
            item.get("tracker_id")
            if "tracker_id" in item
            else item.get("id")
        )

        class_name = (
            item.get("class")
            if "class" in item
            else item.get("class_name")
        )

        court_x = item.get("court_x_m")

        court_y = item.get("court_y_m")

        # Some formats may use x/y directly.
        if court_x is None:
            court_x = item.get("x")

        if court_y is None:
            court_y = item.get("y")

        x = safe_float(court_x)
        y = safe_float(court_y)

        if x is None or y is None:
            continue

        normalized.append(
            {
                "tracker_id": safe_int(tracker_id),
                "class": class_name,
                "court_x_m": x,
                "court_y_m": y,
            }
        )

    return normalized


# ============================================================
# HELPER:
# Get tracker detections
# ============================================================

def get_tracked_detections(prediction):
    """
    Get the tracked_detections output.

    This is useful for debugging and for confirming that
    YOLO + ByteTrack are working.
    """

    if not isinstance(prediction, dict):
        return None

    return prediction.get("tracked_detections")


# ============================================================
# HELPER:
# Fallback player information from ByteTrack
# ============================================================

def get_tracker_players(prediction):

    detections = get_tracked_detections(prediction)

    if detections is None:
        return []

    try:

        count = len(detections)

    except Exception:
        return []

    players = []

    for i in range(count):

        try:

            tracker_id = None

            if detections.tracker_id is not None:
                tracker_id = detections.tracker_id[i]

            xyxy = detections.xyxy[i]

            x1, y1, x2, y2 = xyxy

            # Bottom-center of bounding box.
            foot_x = (float(x1) + float(x2)) / 2.0
            foot_y = float(y2)

            players.append(
                {
                    "tracker_id": safe_int(tracker_id),
                    "pixel_x": foot_x,
                    "pixel_y": foot_y,
                }
            )

        except Exception:
            continue

    return players


# ============================================================
# HELPER:
# Select FAR and NEAR players
# ============================================================

def select_far_near(court_records):
    """
    Determine the two active players.

    Court calibration is:
        near side = y approximately 0
        far side  = y approximately 13.40

    Therefore:
        smaller court_y_m -> near player
        larger court_y_m  -> far player
    """

    if not court_records:
        return None, None

    # Keep only player detections when class is available.
    players = []

    for record in court_records:

        class_name = record.get("class")

        if class_name is None:
            players.append(record)

        elif str(class_name).lower() == "player":
            players.append(record)

    if not players:
        players = court_records

    # We only need the two active players.
    players = sorted(
        players,
        key=lambda item: item["court_y_m"]
    )

    # If more than two objects somehow appear,
    # use the two records with the most separated
    # court positions.
    if len(players) > 2:

        # Usually the two active players are the
        # extreme court-y positions.
        players = [
            players[0],
            players[-1],
        ]

    if len(players) == 1:

        # Only one player visible.
        only_player = players[0]

        # Decide whether it is closer to near/far
        # based on court y.
        if only_player["court_y_m"] < 6.7:
            return None, only_player
        else:
            return only_player, None

    if len(players) >= 2:

        near_player = players[0]
        far_player = players[-1]

        return far_player, near_player

    return None, None


# ============================================================
# FRAME COUNTER
# ============================================================

processed_frames = 0

frames_with_court_coordinates = 0

total_court_player_records = 0


# ============================================================
# MAIN CALLBACK
# ============================================================

def on_prediction(prediction, video_frame):

    global processed_frames
    global frames_with_court_coordinates
    global total_court_player_records

    processed_frames += 1

    # --------------------------------------------------------
    # Get actual frame from VideoFrame
    # --------------------------------------------------------

    try:
        frame = video_frame.image

    except Exception:

        print(
            f"\nCould not read image from VideoFrame "
            f"at frame {processed_frames}"
        )

        return

    if frame is None:
        return

    # --------------------------------------------------------
    # Initialize output video
    # --------------------------------------------------------

    initialize_video_writer(frame)

    # --------------------------------------------------------
    # Extract Workflow court coordinates
    # --------------------------------------------------------

    court_records = extract_court_coordinates(prediction)

    total_court_player_records += len(court_records)

    if len(court_records) > 0:
        frames_with_court_coordinates += 1

    # --------------------------------------------------------
    # Determine FAR and NEAR players
    # --------------------------------------------------------

    far_player, near_player = select_far_near(
        court_records
    )

    # --------------------------------------------------------
    # Tracker debug information
    # --------------------------------------------------------

    tracked_players = get_tracker_players(prediction)

    # --------------------------------------------------------
    # Frame number
    # --------------------------------------------------------

    try:
        frame_number = int(video_frame.frame_id)

    except Exception:
        frame_number = processed_frames

    time_seconds = (
        frame_number / FPS
        if FPS > 0
        else 0.0
    )

    # --------------------------------------------------------
    # DEBUG OUTPUT
    # --------------------------------------------------------

    if frame_number in DEBUG_FRAMES:

        print()
        print("=" * 70)
        print(f"WORKFLOW DEBUG - FRAME {frame_number}")
        print("=" * 70)

        if isinstance(prediction, dict):

            print("Workflow output keys:")
            print(list(prediction.keys()))

            print()

            for key, value in prediction.items():

                print(f"OUTPUT NAME: {key}")
                print(f"TYPE: {type(value)}")

                # Do not print huge image objects.
                if key == "output_image":
                    print("VALUE: <WorkflowImageData>")

                elif key == "tracked_detections":

                    try:
                        print(
                            f"Number of detections: {len(value)}"
                        )

                        if len(value) > 0:

                            print(
                                "xyxy:",
                                value.xyxy
                            )

                            print(
                                "confidence:",
                                value.confidence
                            )

                            print(
                                "class_id:",
                                value.class_id
                            )

                            print(
                                "tracker_id:",
                                value.tracker_id
                            )

                    except Exception as e:

                        print(
                            "Could not inspect tracked_detections:",
                            e
                        )

                elif key == "court_coordinates":

                    print(
                        "COURT COORDINATES:",
                        value
                    )

                else:

                    try:
                        print("VALUE:", value)

                    except Exception:
                        pass

                print()

        print("Parsed court records:")
        print(court_records)

        print()

        print("FAR PLAYER:")
        print(far_player)

        print()

        print("NEAR PLAYER:")
        print(near_player)

        print()

        print("Tracked players:")
        print(tracked_players)

        print("=" * 70)
        print()

    # --------------------------------------------------------
    # CSV values
    # --------------------------------------------------------

    if far_player is not None:

        far_tracker_id = far_player.get("tracker_id")

        far_x = far_player.get("court_x_m")

        far_y = far_player.get("court_y_m")

    else:

        far_tracker_id = None
        far_x = None
        far_y = None

    if near_player is not None:

        near_tracker_id = near_player.get("tracker_id")

        near_x = near_player.get("court_x_m")

        near_y = near_player.get("court_y_m")

    else:

        near_tracker_id = None
        near_x = None
        near_y = None

    # --------------------------------------------------------
    # Write CSV row
    # --------------------------------------------------------

    csv_writer.writerow(
        [
            frame_number,
            round(time_seconds, 4),

            far_tracker_id,
            far_x,
            far_y,

            near_tracker_id,
            near_x,
            near_y,
        ]
    )

    # --------------------------------------------------------
    # Flush periodically so data is not lost
    # --------------------------------------------------------

    if processed_frames % 100 == 0:
        csv_file.flush()

        print(
            f"Processed frame {processed_frames} / "
            f"{TOTAL_FRAMES} | "
            f"court-coordinate frames: "
            f"{frames_with_court_coordinates}"
        )

    # --------------------------------------------------------
    # Write output video
    #
    # We use the frame supplied by InferencePipeline.
    # This guarantees the output video is valid even if
    # output_image has a special WorkflowImageData format.
    # --------------------------------------------------------

    try:

        output_frame = frame.copy()

        # Add a small status overlay.
        status_text = (
            f"Frame: {frame_number} | "
            f"Court players: {len(court_records)}"
        )

        cv2.putText(
            output_frame,
            status_text,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Draw court-coordinate players if available.
        for player in court_records:

            x_m = player["court_x_m"]
            y_m = player["court_y_m"]
            tracker_id = player["tracker_id"]

            text = (
                f"ID {tracker_id}: "
                f"({x_m:.2f}m, {y_m:.2f}m)"
            )

            # We don't know the exact pixel location from
            # court_coordinates alone, so place the text
            # in a readable area.
            y_position = 90 + (
                court_records.index(player) * 35
            )

            cv2.putText(
                output_frame,
                text,
                (30, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        video_writer.write(output_frame)

    except Exception as e:

        print(
            f"Warning: could not write output frame "
            f"{frame_number}: {e}"
        )


# ============================================================
# CREATE INFERENCE PIPELINE
# ============================================================

print()
print("Connecting to Roboflow Workflow...")
print()

pipeline = InferencePipeline.init_with_workflow(
    video_reference=INPUT_VIDEO,

    workspace_name=WORKSPACE,

    workflow_id=WORKFLOW_ID,

    on_prediction=on_prediction,

    api_key=API_KEY,
)


# ============================================================
# RUN PIPELINE
# ============================================================

try:

    print("Connected. Starting workflow...")
    print()

    pipeline.start()

    pipeline.join()

finally:

    # --------------------------------------------------------
    # Close CSV
    # --------------------------------------------------------

    try:
        csv_file.flush()
        csv_file.close()
    except Exception:
        pass

    # --------------------------------------------------------
    # Close video
    # --------------------------------------------------------

    if video_writer is not None:

        try:
            video_writer.release()
        except Exception:
            pass


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("PROCESSING COMPLETE")
print("=" * 70)

print(f"Frames processed              : {processed_frames}")
print(
    f"Frames with court coordinates : "
    f"{frames_with_court_coordinates}"
)
print(
    f"Total court player records    : "
    f"{total_court_player_records}"
)

print()
print(f"CSV saved to:")
print(OUTPUT_CSV)

print()
print(f"Video saved to:")
print(OUTPUT_VIDEO)

print("=" * 70)
print()


# ============================================================
# FINAL VALIDATION
# ============================================================

if frames_with_court_coordinates == 0:

    print()
    print("WARNING")
    print("-" * 70)
    print("No court_coordinates were received.")
    print()
    print(
        "This means the running workflow still did not "
        "return the Court Coordinates Mapper output."
    )
    print()
    print(
        "Check the WORKFLOW DEBUG output above."
    )
    print("-" * 70)

else:

    print()
    print("SUCCESS")
    print("-" * 70)
    print(
        "Court coordinates were successfully received "
        "from the Roboflow workflow."
    )
    print()
    print(
        "The next step will be to inspect the CSV and "
        "build badminton movement-analysis features."
    )
    print("-" * 70)