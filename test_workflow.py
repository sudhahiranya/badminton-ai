import os
from inference import InferencePipeline

API_KEY = os.environ.get("ROBOFLOW_API_KEY")

count = 0

def on_prediction(predictions, video_frame):
    global count

    count += 1

    print("\n========== PREDICTION ==========")
    print("Top-level keys:")
    print(predictions.keys())

    tracked = predictions.get("tracked_detections")

    print("\nTracked detections:")
    print(tracked)

    if tracked is not None:
        print("\nTracked detection data:")
        print(tracked.data)

        print("\nMetadata:")
        print(tracked.metadata)

    print("================================\n")

    # Exit after first prediction without calling pipeline.stop()
    if count >= 1:
        raise SystemExit


pipeline = InferencePipeline.init_with_workflow(
    video_reference="video/badminton_test.mp4",
    workspace_name="venu-sudha",
    workflow_id="badminton-singles-player-tracking-1789540968597",
    api_key=API_KEY,
    on_prediction=on_prediction,
    max_fps=1,
)

pipeline.start()
pipeline.join()