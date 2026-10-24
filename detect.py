"""Real-time face and eye detection using OpenCV Haar Cascade classifiers.

Usage:
    python detect.py                    # live webcam
    python detect.py --source video.mp4 # a video file
    python detect.py --source photo.jpg # a single image
    python detect.py --list-cameras     # list available webcam indexes

Press 'q' or Esc to quit a live window. In --source image mode, the
annotated result is shown once and saved next to the input as
"<name>_detected.jpg".
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_eye.xml"

FACE_COLOR = (255, 120, 0)   # blue-ish, BGR
EYE_COLOR = (0, 220, 60)     # green, BGR


def load_cascades() -> tuple[cv2.CascadeClassifier, cv2.CascadeClassifier]:
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)
    if face_cascade.empty() or eye_cascade.empty():
        sys.exit(
            "Could not load Haar cascade XML files from the OpenCV install "
            f"at:\n  {FACE_CASCADE_PATH}\n  {EYE_CASCADE_PATH}\n"
            "Reinstalling opencv-python usually fixes this."
        )
    return face_cascade, eye_cascade


def _dedupe_overlapping(boxes, iou_thresh=0.3):
    """detectMultiScale occasionally returns two overlapping boxes for the
    same face at neighboring scales (seen in testing: a 38x38 and an 85x85
    box both centered on the same person). Keep the larger box of any pair
    that overlaps significantly rather than drawing both."""
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept = []
    for bx, by, bw, bh in boxes:
        b_area = bw * bh
        overlaps = False
        for kx, ky, kw, kh in kept:
            ix = max(0, min(bx + bw, kx + kw) - max(bx, kx))
            iy = max(0, min(by + bh, ky + kh) - max(by, ky))
            inter = ix * iy
            union = b_area + kw * kh - inter
            if union > 0 and inter / union > iou_thresh:
                overlaps = True
                break
        if not overlaps:
            kept.append((bx, by, bw, bh))
    return kept


def detect_and_annotate(frame, face_cascade, eye_cascade):
    """Detect faces, then search for eyes only inside each face region —
    scanning the whole frame for eyes produces far more false positives.

    Eye detection is drawn when found but does NOT gate whether a face is
    accepted. An earlier version required at least one eye match before
    accepting a face, meant to reject a false positive that appeared on a
    plaid shirt during testing — but verified against three real test
    photos (a side-angle single face, a two-person photo, and a group
    photo), that gate rejected 3/3 legitimate faces: small or angled face
    crops frequently have no cleanly-detectable eye. Loosening the eye
    parameters enough to catch those real eyes also let the eye cascade
    match the same plaid fabric that fooled the face cascade.

    A second, independent false positive later showed up on a striped
    football jersey in an unrelated test photo — same failure mode on a
    different repeating texture. Two occurrences on unrelated images is
    strong evidence this is a genuine, reproducible limitation of Haar
    cascades on high-frequency textures, not tunable away with stricter
    minNeighbors (tested up to 12 with no change) or eye-gating (tested
    above). It is a known, documented trade-off of the classical approach
    — see README — rather than something hidden or silently "fixed" by
    breaking real detections instead.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Scale the minimum face size to the frame instead of a fixed pixel
    # value, so this works on a tight portrait crop, a wide group photo,
    # and a 640x480 webcam frame without separate tuning for each.
    min_face = max(30, int(min(w, h) * 0.08))

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=6, minSize=(min_face, min_face)
    )
    faces = _dedupe_overlapping(faces)
    eye_count = 0

    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), FACE_COLOR, 2)

        face_roi_gray = gray[fy : fy + fh, fx : fx + fw]
        face_roi_color = frame[fy : fy + fh, fx : fx + fw]
        min_eye = max(10, int(fw * 0.12))

        eyes = eye_cascade.detectMultiScale(
            face_roi_gray, scaleFactor=1.05, minNeighbors=4,
            minSize=(min_eye, min_eye),
        )
        # Eyes sit in the upper ~60% of a face; discard anything lower
        # (nostrils, mouth corners) that the classifier occasionally flags.
        for (ex, ey, ew, eh) in eyes:
            if ey + eh / 2 > fh * 0.6:
                continue
            cv2.rectangle(
                face_roi_color, (ex, ey), (ex + ew, ey + eh), EYE_COLOR, 2
            )
            eye_count += 1

    return frame, len(faces), eye_count


def run_on_image(path: Path, face_cascade, eye_cascade):
    frame = cv2.imread(str(path))
    if frame is None:
        sys.exit(f"Could not read image: {path}")

    annotated, face_count, eye_count = detect_and_annotate(frame, face_cascade, eye_cascade)
    out_path = path.with_name(f"{path.stem}_detected{path.suffix}")
    cv2.imwrite(str(out_path), annotated)
    print(f"Detected {face_count} face(s), {eye_count} eye(s). Saved: {out_path}")

    cv2.imshow("Face & Eye Detection - press any key to close", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_on_stream(source, face_cascade, eye_cascade):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(
            f"Could not open video source: {source!r}\n"
            "If this is a webcam index, try --list-cameras to see what's available, "
            "or check camera permissions for your terminal/IDE in System Settings."
        )

    prev_time = time.time()
    fps = 0.0

    print("Press 'q' or Esc to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Stream ended or camera disconnected.")
                break

            frame, face_count, eye_count = detect_and_annotate(frame, face_cascade, eye_cascade)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
            prev_time = now

            cv2.putText(
                frame, f"FPS: {fps:.1f}  Faces: {face_count}  Eyes: {eye_count}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
            )
            cv2.imshow("Face & Eye Detection - press q to quit", frame)

            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def list_cameras(max_index: int = 5) -> None:
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            found.append(i)
        cap.release()
    print(f"Available camera indexes: {found or 'none found'}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", default="0",
        help="Webcam index (default 0), or a path to a video/image file.",
    )
    parser.add_argument(
        "--list-cameras", action="store_true",
        help="List available webcam indexes and exit.",
    )
    return parser.parse_args()


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    args = parse_args()

    if args.list_cameras:
        list_cameras()
        return

    face_cascade, eye_cascade = load_cascades()

    candidate_path = Path(args.source)
    if candidate_path.suffix.lower() in IMAGE_EXTS and candidate_path.exists():
        run_on_image(candidate_path, face_cascade, eye_cascade)
        return

    # Webcam index if numeric, otherwise treat as a video file path
    source = int(args.source) if args.source.isdigit() else args.source
    run_on_stream(source, face_cascade, eye_cascade)


if __name__ == "__main__":
    main()
