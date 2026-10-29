```python
"""Flask web application for real-time face and eye detection.

The browser captures webcam frames and sends them to Flask for processing.
OpenCV detects faces and eyes using the shared implementation in detect.py.

Routes:
    GET  /               Main page
    POST /start          Starts detection
    POST /stop           Stops detection
    POST /reset          Resets face and eye counters
    GET  /status          Returns detection status and counts
    POST /process_frame  Processes a webcam frame and returns annotated JPEG
"""

import threading

import cv2
import numpy as np

from flask import Flask, Response, jsonify, render_template, request

from detect import detect_and_annotate, load_cascades


app = Flask(__name__)

# Limit incoming requests to 2 MB.
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

# Load Haar Cascade classifiers once when the application starts.
face_cascade, eye_cascade = load_cascades()

# Protect shared detection state and counters.
state_lock = threading.Lock()

detection_active = False
latest_counts = {
    "faces": 0,
    "eyes": 0
}


@app.route("/")
def index():
    """Display the main detection page."""
    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start_detection():
    """Activate detection."""
    global detection_active

    with state_lock:
        detection_active = True

    return jsonify(status="started")


@app.route("/stop", methods=["POST"])
def stop_detection():
    """Deactivate detection."""
    global detection_active

    with state_lock:
        detection_active = False

    return jsonify(status="stopped")


@app.route("/reset", methods=["POST"])
def reset_counts():
    """Reset the face and eye counters."""
    global latest_counts

    with state_lock:
        latest_counts = {
            "faces": 0,
            "eyes": 0
        }

    return jsonify(status="reset")


@app.route("/status", methods=["GET"])
def status():
    """Return the current detection state and counters."""
    with state_lock:
        return jsonify(
            active=detection_active,
            faces=latest_counts["faces"],
            eyes=latest_counts["eyes"]
        )


@app.route("/process_frame", methods=["POST"])
def process_frame():
    """Process a webcam frame sent by the browser.

    Expects:
        Multipart form data containing a file named 'frame'.

    Returns:
        Annotated JPEG image.
        X-Face-Count and X-Eye-Count response headers.
    """
    global latest_counts

    # Check whether detection is active.
    with state_lock:
        if not detection_active:
            return jsonify(error="Detection is not active"), 400

    # Validate the uploaded frame.
    if "frame" not in request.files:
        return jsonify(error="No frame received"), 400

    image_bytes = request.files["frame"].read()

    if not image_bytes:
        return jsonify(error="Empty image received"), 400

    # Convert the uploaded bytes into an OpenCV image.
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify(error="Invalid image format"), 400

    # Protect against excessively large decoded images.
    height, width = frame.shape[:2]

    if width * height > 4_000_000:
        return jsonify(error="Image dimensions are too large"), 400

    try:
        # Reuse the project's existing detection implementation.
        annotated_frame, face_count, eye_count = detect_and_annotate(
            frame,
            face_cascade,
            eye_cascade
        )

        # Encode the annotated frame as JPEG.
        success, encoded_image = cv2.imencode(
            ".jpg",
            annotated_frame,
            [cv2.IMWRITE_JPEG_QUALITY, 75]
        )

        if not success:
            return jsonify(error="Failed to encode image"), 500

    except Exception:
        app.logger.exception("Frame processing failed")
        return jsonify(error="Frame processing failed"), 500

    # Update the counters only if detection is still active.
    with state_lock:
        if not detection_active:
            return jsonify(error="Detection has stopped"), 400

        latest_counts = {
            "faces": int(face_count),
            "eyes": int(eye_count)
        }

    # Return the annotated image and detection counts.
    response = Response(
        encoded_image.tobytes(),
        mimetype="image/jpeg"
    )

    response.headers["X-Face-Count"] = str(int(face_count))
    response.headers["X-Eye-Count"] = str(int(eye_count))
    response.headers["Cache-Control"] = "no-store"

    return response


@app.errorhandler(413)
def request_too_large(error):
    """Handle uploads exceeding the configured request-size limit."""
    return jsonify(error="Uploaded frame exceeds the size limit"), 413


if __name__ == "__main__":
    # Local development server.
    # Azure should use Gunicorn instead.
    app.run(
        host="127.0.0.1",
        port=5001,
        debug=False,
        threaded=True
    )
```
