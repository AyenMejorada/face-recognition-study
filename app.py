"""Webcam emotion recognition demo.

Detects faces with an OpenCV Haar cascade and labels the dominant emotion on
each face using DeepFace. Press "q" to quit.

This is the script version of the webcam cell from Code.ipynb.
"""

import cv2
from deepface import DeepFace

# Run DeepFace only on every Nth frame instead of every frame. Emotion analysis
# is slow, so analyzing every frame makes the video feel laggy. Between analyses
# we reuse the last label we computed, which keeps the display smooth.
ANALYZE_EVERY_N_FRAMES = 5

# Load the cascade that ships with the OpenCV package itself. Using
# cv2.data.haarcascades means we don't depend on the XML file living in the repo
# and the path works on any machine where opencv-python is installed.
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Define the font once so we aren't recreating it inside the draw loop.
FONT = cv2.FONT_HERSHEY_SIMPLEX


def open_camera():
    """Open the webcam, trying index 0 first and falling back to index 1.

    Different machines expose the default camera on different indices, so we try
    the most common one first and fall back before giving up. We raise a clear
    error if neither opens so the caller knows what went wrong.
    """
    for index in (0, 1):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            return cap
        # Release the failed handle before trying the next index so we don't
        # leak the device.
        cap.release()

    raise IOError("Cannot open webcam (tried camera indices 0 and 1)")


def analyze_emotion(face_crop):
    """Return the dominant emotion for a single face crop.

    We analyze just the cropped face (not the whole frame) so DeepFace focuses
    on the region the cascade already found. enforce_detection=False stops
    DeepFace from raising when it isn't fully confident, which keeps the live
    loop from crashing on a bad frame.
    """
    result = DeepFace.analyze(
        face_crop,
        actions=["emotion"],
        enforce_detection=False,
    )
    # DeepFace returns a list of results (one per detected face). Since we pass a
    # single face crop, we read the first entry.
    return result[0]["dominant_emotion"]


def annotate_frame(frame, faces, labels):
    """Draw a box and emotion label for each detected face.

    `faces` is the list of (x, y, w, h) boxes and `labels` holds the matching
    emotion text for each box.
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


def main():
    """Run the live webcam emotion recognition loop."""
    cap = open_camera()

    # Remember the last label we computed per face so the frames we skip can
    # still show something sensible instead of a blank box.
    last_labels = []
    frame_count = 0

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

            # Only run the expensive DeepFace analysis every Nth frame. On the
            # other frames we reuse last_labels.
            if frame_count % ANALYZE_EVERY_N_FRAMES == 0:
                labels = []
                for (x, y, w, h) in faces:
                    # Crop the face region and analyze just that patch.
                    face_crop = frame[y:y + h, x:x + w]
                    labels.append(analyze_emotion(face_crop))
                last_labels = labels
            else:
                labels = last_labels

            # If the number of detected faces changed on a skipped frame, the
            # cached labels may not line up. Pad or trim so every box still gets
            # some text to draw.
            if len(labels) < len(faces):
                labels = labels + [""] * (len(faces) - len(labels))

            annotate_frame(frame, faces, labels)

            cv2.imshow("Emotion Recognition Camera", frame)

            # waitKey keeps the window responsive; quit when "q" is pressed.
            if cv2.waitKey(2) & 0xFF == ord("q"):
                break

            frame_count += 1
    finally:
        # Always release the camera and close the window, even if something
        # above throws, so the device isn't left locked.
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
