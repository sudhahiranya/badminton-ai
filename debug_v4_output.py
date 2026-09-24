import os
from inference import InferencePipeline

WORKSPACE = "venu-sudha"
WORKFLOW_ID = "badminton-singles-players-vbadminton-singles-players-2-yolo26n-t1-logic"
INPUT_VIDEO = "video/badminton_test.mp4"

pipeline = None
seen = False


def on_prediction(prediction, video_frame):
    global pipeline, seen

    if seen:
        return

    seen = True

    print()
    print("=" * 70)
    print("WORKFLOW OUTPUT DEBUG")
    print("=" * 70)

    print("Prediction type:", type(prediction))

    if isinstance(prediction, dict):
        print("Top-level keys:", list(prediction.keys()))

        for key, value in prediction.items():
            print()
            print(f"KEY: {key}")
            print(f"TYPE: {type(value)}")

            if isinstance(value, dict):
                print("DICT KEYS:", list(value.keys()))

                for subkey, subvalue in value.items():
                    print(
                        f"  {subkey}: "
                        f"{type(subvalue).__name__}"
                    )

            elif isinstance(value, list):
                print("LIST LENGTH:", len(value))

                if len(value) > 0:
                    print(
                        "FIRST ITEM TYPE:",
                        type(value[0])
                    )
                    print(
                        "FIRST ITEM:",
                        repr(value[0])[:1500]
                    )

            else:
                print("VALUE:", repr(value)[:1500])

    else:
        print("Prediction repr:")
        print(repr(prediction)[:5000])

    print()
    print("=" * 70)
    print("STOPPING AFTER FIRST RESULT")
    print("=" * 70)

    pipeline.terminate()


pipeline = InferencePipeline.init_with_workflow(
    video_reference=INPUT_VIDEO,
    workspace_name=WORKSPACE,
    workflow_id=WORKFLOW_ID,
    on_prediction=on_prediction,
)

print("Starting diagnostic...")
pipeline.start()
pipeline.join()

print("Diagnostic finished.")