import cv2

INPUT = "video/badminton_test.mp4"
OUTPUT = "video/debug_test.mp4"

cap = cv2.VideoCapture(INPUT)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    OUTPUT,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

max_frames = int(fps * 7)

for i in range(max_frames):
    ret, frame = cap.read()

    if not ret:
        break

    writer.write(frame)

cap.release()
writer.release()

print(f"Created {OUTPUT}")
print(f"Frames: {i + 1}")