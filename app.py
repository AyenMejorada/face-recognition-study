"""Webcam emotion recognition demo.

Detects faces with an OpenCV Haar cascade and labels the dominant emotion on
each face using DeepFace. Labels are smoothed over the last few frames and show
a confidence percentage.

Keyboard shortcuts while the window is focused:
  q  quit
  s  save the current annotated frame to screenshots/

This is the script version of the webcam cell from Code.ipynb.
"""

import argparse
import os
import time
from collections import deque

import cv2

# Note: DeepFace is imported lazily (inside analyze_emotion) rather than here.
# Importing it pulls in TensorFlow, which takes several seconds. Keeping it out
# of the top level means "python app.py --help" stays instant.

# Run DeepFace only on every Nth frame instead of every frame. Emotion analysis
# is slow, so analyzing every frame makes the video feel laggy. Between analyses
# we reuse the smoothed labels we already have. This is the default for --every.
ANALYZE_EVERY_N_FRAMES = 5

# How many recent readings to average per face. Averaging stops the label from
# flickering between emotions frame to frame. Default for --smooth.
SMOOTHING_WINDOW = 5

# If a detected box's center is within this many pixels of a previous box's
# center, we treat them as the same face. This keeps a face's history attached
# to it as it moves a little between frames.
MATCH_DISTANCE = 80

# Drop a tracked face if we haven't seen it for this many frames (it left the
# view). Stops stale histories from lingering.
MAX_MISSING_FRAMES = 15

# Folder where the "s" key saves annotated screenshots.
SCREENSHOT_DIR = "screenshots"

# Load the cascade that ships with the OpenCV package itself. Using
# cv2.data.haarcascades means we don't depend on the XML file living in the repo
# and the path works on any machine where opencv-python is installed.
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Define the font once so we aren't recreating it inside the draw loop.
FONT = cv2.FONT_HERSHEY_SIMPLEX


def open_camera(preferred=0):
    """Open the webcam, trying `preferred` first and falling back to the other.

    Different machines expose the default camera on different indices, so we try
    the preferred one first and fall back to the other common index (0/1) before
    giving up. We raise a clear error if neither opens so the caller knows what
    went wrong.
    """
    fallback = 1 if preferred == 0 else 0
    for index in (preferred, fallback):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            return cap
        # Release the failed handle before trying the next index so we don't
        # leak the device.
        cap.release()

    raise IOError(
        f"Cannot open webcam (tried camera indices {preferred} and {fallback})"
    )


def analyze_emotion(face_crop):
    """Return the per-emotion score dict for a single face crop.

    We analyze just the cropped face (not the whole frame) so DeepFace focuses
    on the region the cascade already found. enforce_detection=False stops
    DeepFace from raising when it isn't fully confident, which keeps the live
    loop from crashing on a bad frame. We return the full score dict (e.g.
    {"happy": 87.0, "sad": 2.1, ...}) so the caller can smooth the scores.
    """
    # Imported here (not at the top) so loading TensorFlow is deferred until we
    # actually analyze a frame. This keeps "python app.py --help" instant.
    from deepface import DeepFace

    result = DeepFace.analyze(
        face_crop,
        actions=["emotion"],
        enforce_detection=False,
    )
    # DeepFace returns a list of results (one per detected face). Since we pass a
    # single face crop, we read the first entry.
    return result[0]["emotion"]


def box_center(box):
    """Return the (cx, cy) center point of an (x, y, w, h) box."""
    x, y, w, h = box
    return (x + w / 2, y + h / 2)


def match_faces(faces, tracks, next_id, smooth):
    """Match current detections to existing tracks by nearest center.

    `tracks` maps a track id to a dict with its last box, score history, and how
    many frames it has been missing. We greedily match each detected box to the
    closest unused track within MATCH_DISTANCE; anything left over becomes a new
    track. Returns a list of track ids (one per detected box) and the updated
    next_id counter.

    A simple nearest-center match is enough here because faces move only a
    little between frames, and it keeps each face's smoothing history attached
    to it as it drifts around.
    """
    assigned = []
    used_track_ids = set()

    for box in faces:
        cx, cy = box_center(box)
        best_id = None
        best_dist = MATCH_DISTANCE

        for track_id, track in tracks.items():
            if track_id in used_track_ids:
                continue
            tcx, tcy = box_center(track["box"])
            dist = ((cx - tcx) ** 2 + (cy - tcy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_id = track_id

        if best_id is None:
            # No nearby existing face, so start a new track. The history length
            # comes from --smooth so the window size is configurable.
            best_id = next_id
            next_id += 1
            tracks[best_id] = {
                "box": box,
                "history": deque(maxlen=smooth),
                "missing": 0,
            }

        used_track_ids.add(best_id)
        assigned.append(best_id)

    return assigned, next_id


def smoothed_label(history):
    """Return a "emotion NN%" label from a face's score history.

    `history` is a deque of per-emotion score dicts. We average each emotion's
    score across the window and pick the highest average. Averaging is what
    stops the label from jumping around frame to frame.
    """
    if not history:
        return ""

    # Sum each emotion across the recorded frames, then divide by the count.
    totals = {}
    for scores in history:
        for emotion, value in scores.items():
            totals[emotion] = totals.get(emotion, 0.0) + value

    count = len(history)
    best_emotion = max(totals, key=totals.get)
    best_avg = totals[best_emotion] / count

    # DeepFace scores are already on a 0-100 scale, so round to a whole percent.
    return f"{best_emotion} {round(best_avg)}%"


def annotate_frame(frame, faces, labels):
    """Draw a box and emotion label for each detected face.

    `faces` is the list of (x, y, w, h) boxes and `labels` holds the matching
    label text for each box.
    """
    for (x, y, w, h), label in zip(faces, labels):
        # Green box around the face.
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Prefer drawing the label just above the box. If the box sits too close
        # to the top edge there isn't room above it, so draw the label inside
        # the box instead to keep it visible.
        if y - 10 > 10:
            label_pos = (x, y - 10)
        else:
            label_pos = (x, y + 20)

        cv2.putText(frame, label, label_pos, FONT, 0.9, (255, 0, 0), 2, cv2.LINE_4)

    return frame


def draw_hint(frame):
    """Draw the keyboard hint along the bottom of the window."""
    height = frame.shape[0]
    cv2.putText(
        frame,
        "q: quit | s: save",
        (10, height - 10),
        FONT,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def save_screenshot(frame):
    """Save `frame` to the screenshots folder with a timestamped name.

    Creates the folder if needed and prints the saved path so the user can find
    the file. Returns the path that was written.
    """
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    filename = time.strftime("emotion_%Y%m%d_%H%M%S.png")
    path = os.path.join(SCREENSHOT_DIR, filename)
    cv2.imwrite(path, frame)
    print(f"Saved screenshot: {path}")
    return path


def parse_args():
    """Parse command-line options. Defaults keep the old no-argument behavior."""
    parser = argparse.ArgumentParser(
        description="Webcam emotion recognition demo (OpenCV + DeepFace)."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Preferred camera index (default: 0). Falls back to the other "
        "index if it fails to open.",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=ANALYZE_EVERY_N_FRAMES,
        help="Analyze every Nth frame with DeepFace (default: "
        f"{ANALYZE_EVERY_N_FRAMES}). Lower is more responsive but slower.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=SMOOTHING_WINDOW,
        help="How many recent readings to average per face (default: "
        f"{SMOOTHING_WINDOW}). Larger is steadier but slower to react.",
    )
    return parser.parse_args()


def main():
    """Run the live webcam emotion recognition loop."""
    args = parse_args()

    cap = open_camera(args.camera)

    # tracks maps a track id -> {"box", "history" (deque of score dicts),
    # "missing"}. We keep per-face history so each label can be smoothed
    # independently as faces move around.
    tracks = {}
    next_id = 0
    frame_count = 0
    prev_face_count = 0

    try:
        while True:
            ret, frame = cap.read()
            # If the read failed, ret is False and the frame is unusable. Break
            # out cleanly rather than trying to process a bad frame.
            if not ret:
                break

            # Haar cascades work on grayscale images, so convert first.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 4)

            # Match this frame's boxes to existing tracks so each face keeps its
            # own score history as it moves.
            assigned_ids, next_id = match_faces(faces, tracks, next_id, args.smooth)

            # A newly tracked face starts with an empty history, so it would
            # show no label until the next Nth frame. Analyze right away if any
            # detected face has no readings yet. This covers the case where one
            # face leaves and another enters in the same frame (the face count
            # is unchanged, so the count check below wouldn't catch it).
            has_new_face = any(
                not tracks[track_id]["history"] for track_id in assigned_ids
            )

            # Decide whether to run the expensive analysis this frame. Normally
            # we only run it every Nth frame, but if the number of faces changed
            # (someone entered or left) we analyze right away so labels don't
            # stick to the wrong face.
            face_count_changed = len(faces) != prev_face_count
            should_analyze = (
                frame_count % args.every == 0 or face_count_changed or has_new_face
            )

            if should_analyze:
                for box, track_id in zip(faces, assigned_ids):
                    x, y, w, h = box
                    face_crop = frame[y:y + h, x:x + w]
                    scores = analyze_emotion(face_crop)
                    tracks[track_id]["history"].append(scores)

            # Update each matched track's last-seen box and reset its missing
            # counter.
            for box, track_id in zip(faces, assigned_ids):
                tracks[track_id]["box"] = box
                tracks[track_id]["missing"] = 0

            # Age out tracks we didn't see this frame; drop ones gone too long.
            seen = set(assigned_ids)
            for track_id in list(tracks.keys()):
                if track_id not in seen:
                    tracks[track_id]["missing"] += 1
                    if tracks[track_id]["missing"] > MAX_MISSING_FRAMES:
                        del tracks[track_id]

            # Build the smoothed label for each currently detected face.
            labels = [
                smoothed_label(tracks[track_id]["history"])
                for track_id in assigned_ids
            ]

            annotate_frame(frame, faces, labels)
            draw_hint(frame)

            cv2.imshow("Emotion Recognition Camera", frame)

            # waitKey keeps the window responsive. Read one key per frame.
            key = cv2.waitKey(2) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                # Save the frame as shown (boxes, labels, and hint).
                save_screenshot(frame)

            prev_face_count = len(faces)
            frame_count += 1
    finally:
        # Always release the camera and close the window, even if something
        # above throws, so the device isn't left locked.
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
