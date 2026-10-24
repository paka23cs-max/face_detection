"""Flask web app for real-time face and eye detection.

This reuses detect_and_annotate() and load_cascades() from detect.py (the
CLI version of this project) so the web app and the CLI tool share one
tested detection implementation instead of two copies of the same logic.

Routes:
    GET  /                Main page (templates/index.html)
    GET  /video_feed      MJPEG video stream with detection boxes drawn on it
    POST /start           Opens the webcam and starts the detection loop
    POST /stop            Stops the loop and releases the webcam
    POST /reset           Resets the face/eye counters back to zero
    GET  /status          JSON: {active, faces, eyes} - polled by script.js
"""

import threading

import cv2
from flask import Flask, Response, jsonify, render_template

from detect import detect_and_annotate, load_cascades

app = Flask(__name__)

face_cascade, eye_cascade = load_cascades()

# Flask can handle multiple requests concurrently (threaded=True below),
# so a lock protects the camera and counters from being touched by two
# requests (e.g. the video stream and a /stop click) at the same instant.
state_lock = threading.Lock()
camera = None
detection_active = False
latest_counts = {"faces": 0, "eyes": 0}


def open_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    return camera


def close_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None


def generate_frames():
    """Yields one MJPEG-framed JPEG per detected frame until detection is
    stopped. This is the generator behind the /video_feed route."""
    global detection_active, latest_counts

    while True:
        with state_lock:
            if not detection_active:
                break
            cam = open_camera()
            ok, frame = cam.read()

            if not ok:
                # Camera disconnected or never opened (e.g. no webcam
                # permission) - stop cleanly instead of leaving the
                # frontend stuck showing "Active" with a dead stream.
                detection_active = False
                close_camera()
                break

        frame, face_count, eye_count = detect_and_annotate(frame, face_cascade, eye_cascade)
        latest_counts = {"faces": face_count, "eyes": eye_count}

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        # multipart/x-mixed-replace: each part replaces the previous one in
        # the browser's <img> tag, which is what makes this look like video.
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/start", methods=["POST"])
def start_detection():
    global detection_active
    with state_lock:
        detection_active = True
        open_camera()
    return jsonify(status="started")


@app.route("/stop", methods=["POST"])
def stop_detection():
    global detection_active
    with state_lock:
        detection_active = False
        close_camera()
    return jsonify(status="stopped")


@app.route("/reset", methods=["POST"])
def reset_counts():
    global latest_counts
    with state_lock:
        latest_counts = {"faces": 0, "eyes": 0}
    return jsonify(status="reset")


@app.route("/status")
def status():
    with state_lock:
        return jsonify(active=detection_active, **latest_counts)


if __name__ == "__main__":
    # Port 5000 is taken by macOS's AirPlay Receiver (ControlCenter) on many
    # Macs, so this project uses 5001 instead.
    app.run(debug=True, threaded=True, port=5001)
