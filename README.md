# Real-Time Face & Eye Detection

Real-time face and eye detection from a webcam feed, using OpenCV's Haar
Cascade classifiers — applying classical AI/ML computer-vision techniques
to detect faces and eyes live, the kind of pipeline used in security
systems, facial recognition, and human-computer interaction.

This repository has two interfaces built on the same detection code:

- **`detect.py`** — a command-line tool (live webcam, video file, or
  single image).
- **`app.py`** — a Flask web app with a browser UI (Start/Stop/Reset
  controls, live face/eye counts). Built as a 2nd-year B.Tech academic
  project titled *"Real-Time Face and Eye Detection Using Haar Cascade
  Classifiers and OpenCV."*

Both call the exact same `detect_and_annotate()` function from
`detect.py`, so there is one tested detection implementation, not two.

## Objective

Build a real-time computer vision application that:

1. Accesses a webcam.
2. Detects human faces in each frame.
3. Detects eyes within each detected face.
4. Draws bounding boxes around detected faces and eyes.
5. Displays the live detection result with running face/eye counts.
6. Lets the user start and stop detection.
7. Provides a simple web frontend for interacting with it.

## Features

- Real-time face detection via `haarcascade_frontalface_default.xml`.
- Real-time eye detection via `haarcascade_eye.xml`, searched only inside
  each detected face region.
- Live face/eye counts and a detection status indicator.
- Start / Stop / Reset controls in the browser.
- A separate CLI mode that also supports a video file or a single image
  as input, for testing without a live camera.

## Technology stack

| Layer | Technology |
|---|---|
| Computer vision | Python, OpenCV (`cv2`), Haar Cascade classifiers |
| Backend (web app) | Flask |
| Frontend (web app) | HTML5, CSS3, vanilla JavaScript |
| CLI tool | Python, argparse |

Haar Cascade is a **classical, pre-trained** machine-learning object
detector — not a deep-learning model. See [How it works](#how-it-works)
and [AI/ML notes](#aiml-notes-what-is-and-isnt-machine-learning-here)
below for what that distinction means.

## How it works

1. Each frame is converted to grayscale and histogram-equalized, which
   makes Haar features more consistent under uneven lighting.
2. `haarcascade_frontalface_default.xml` finds face regions in the frame.
3. `haarcascade_eye.xml` searches only *inside* each detected face for
   eyes — scanning the whole frame for eyes produces far more false
   positives than constraining the search to a known face region.
4. Detections below face regions (nostrils, mouth corners the eye
   classifier sometimes misfires on) are filtered by position.
5. Matches are drawn as bounding boxes and shown live, with an FPS
   counter so you can see the pipeline is keeping up with the camera.

## Architecture (web app)

```
Browser (HTML/CSS/JS)
   |  loads the page, requests /video_feed, polls /status every 1s
   v
Flask backend (app.py)
   |  opens the webcam via OpenCV
   v
OpenCV frame capture
   |  BGR -> grayscale -> equalizeHist
   v
Haar Cascade face detection (detectMultiScale)
   |  for each detected face:
   v
Haar Cascade eye detection, searched only inside that face's region
   |  boxes drawn on the frame; face/eye counts updated
   v
Frame encoded as JPEG -> streamed back to the browser as MJPEG
   (multipart/x-mixed-replace), displayed live in an <img> tag
```

Data moves in two independent channels: the `<img>` tag holds one open
HTTP connection to `/video_feed` and keeps receiving new JPEG frames; a
separate `fetch()` call in `script.js` polls `/status` once a second for
the current face/eye counts as JSON. Start/Stop/Reset buttons send a
`POST` to `/start`, `/stop`, or `/reset`, which just flip a flag and
open/close the camera - the actual video keeps flowing through
`/video_feed` independently of those calls.

## Folder structure (web app)

```
face-eye-detection/
├── app.py                  # Flask backend: routes, camera loop, MJPEG stream
├── detect.py                # CLI tool + the shared detection function
├── requirements.txt
├── README.md
├── templates/
│   └── index.html           # main page
├── static/
│   ├── css/style.css
│   └── js/script.js          # Start/Stop/Reset logic, status polling
└── screenshots/              # for your report/presentation
```

The Haar cascade XML files themselves aren't copied into this repo -
`load_cascades()` in `detect.py` loads them straight from
`cv2.data.haarcascades`, the path where OpenCV installs its own bundled
copies. That keeps one copy of the XML files instead of two to keep in
sync.

## Run it

### CLI tool

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python detect.py                     # live webcam (default index 0)
python detect.py --source 1          # a different camera index
python detect.py --source clip.mp4   # run on a video file instead
python detect.py --source photo.jpg  # run on a single image
python detect.py --list-cameras      # see which camera indexes exist
```

Press **q** or **Esc** to close the live window. Image mode saves the
annotated result next to the input as `<name>_detected.jpg`.

### Web app

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python app.py
```

Then open **http://localhost:5001** in a browser and click **Start
Detection**. (Port 5001, not 5000 - on macOS, port 5000 is often taken
by AirPlay Receiver / ControlCenter.)

### Troubleshooting the webcam

- **macOS**: the first time you run either the CLI or `app.py`, macOS
  will prompt for camera access for your terminal/IDE. If you don't see
  a prompt or detection never starts, check System Settings → Privacy &
  Security → Camera and enable it for your terminal app (Terminal,
  iTerm, VS Code, etc.), then re-run.
- **Camera already in use**: close other apps that might be holding the
  webcam (Zoom, FaceTime, another browser tab).
- **Wrong camera selected**: run `python detect.py --list-cameras` to
  see which index is your actual webcam if you have more than one
  (e.g. a laptop camera plus a USB webcam).
- **Web app shows "Stopped" right after clicking Start**: the backend
  couldn't read a frame from the camera (permission not granted, or no
  camera present) and automatically stopped itself rather than showing
  a frozen video - check the terminal running `app.py` for the OpenCV
  error message.

## AI/ML notes: what is (and isn't) machine learning here

- **Where ML is used**: the Haar Cascade classifiers themselves. A Haar
  Cascade is trained using the **Viola-Jones algorithm** on thousands of
  labeled *positive samples* (cropped faces or eyes) and *negative
  samples* (images with no face/eye) to learn which simple rectangular
  light/dark patterns ("Haar features") best separate the two, arranged
  into stages that reject obvious non-matches early for speed.
- **This project uses a pretrained classifier, not a trained one.** The
  XML files (`haarcascade_frontalface_default.xml`,
  `haarcascade_eye.xml`) ship with OpenCV, already trained by OpenCV's
  developers years ago. No training happens in this project or its code
  - `detectMultiScale()` only *runs* that pretrained model against a new
  frame.
- **Why this is "classical" ML, not deep learning**: Haar Cascades use
  hand-engineered features (fixed rectangular patterns) and a shallow
  cascade of simple classifiers, not a deep neural network that learns
  its own features from raw pixels through many layers (like a CNN).
  Haar Cascades are much faster and lighter, but less accurate on angled
  faces, poor lighting, or occlusions than modern CNN-based detectors.

## Tested against

Beyond a live webcam, `detect_and_annotate()` was run against seven still
photos — sourced from OpenCV's own official test corpus plus one personal
photo — to check it generalizes past one face in good lighting:

| Image | Result |
|---|---|
| Single portrait (own photo) | Face found. 1 false positive on a plaid shirt lower in the frame. |
| [`messi5.jpg`](https://github.com/opencv/opencv/blob/master/samples/data/messi5.jpg) — single face at an angle, mid-action | Face found. 1 false positive on a striped jersey/blurred background. |
| `karen-and-rob.png` — two people | Both faces found, no false positives. |
| A seven-person group photo | All 7 faces found, no false positives. |
| `audrybt1.png` — single portrait against heavily patterned wallpaper | Face found, no false positives — texture false-positives aren't universal. |
| `er.png` — six-person cast photo, studio lighting | 5 of 6 real faces found; the 6th (seated, angled) was missed, and a false positive landed on a badge/lanyard instead. |
| `churchill-downs.png` — architectural photo, no people | Correctly 0 detections — confirms it doesn't hallucinate faces on columns, roofline, or signage text. |

Net across all seven: **17 of 18** real faces found, **3** false positives —
all three on repeating fabric/lanyard patterns, none on architecture or
plain backgrounds.

## Limitations

- Needs a webcam and, on macOS, camera permission granted to your
  terminal or IDE (System Settings → Privacy & Security → Camera).
- **Haar cascades reliably false-positive on repeating high-frequency
  textures** — confirmed independently on two unrelated test photos, both
  times on patterned clothing (a plaid shirt, a striped jersey). This
  isn't fixable by tuning `minNeighbors` (tested up to 12, no change) or
  by requiring an eye match inside the box before accepting a face — that
  was tried, and rejected 3/3 real faces in testing because small or
  angled faces often have no cleanly-detectable eye, while loosening the
  eye parameters enough to catch those real eyes let the eye cascade match
  the same fabric that fooled the face cascade. It's a genuine limitation
  of the classical approach, not a bug in this script — deep-learning
  detectors (MTCNN, MediaPipe, YuNet) handle this far better, at the cost
  of a much heavier dependency.
- Accuracy drops on side-facing/angled faces, poor or uneven lighting,
  and partial occlusions (hands, masks).
- Eyeglasses can interfere with eye detection, since glare or thick
  frames can hide the features the eye cascade looks for.
- Detection quality depends on webcam resolution and frame quality.
- **Requires OpenCV 4.x.** OpenCV 5.0 removed `cv2.CascadeClassifier`
  and no longer bundles the Haar cascade XML files at all, in favor of
  the newer DNN-based `FaceDetectorYN`. `requirements.txt` pins
  `opencv-python<5` for that reason — installing latest will break this
  script.
- The web app's MJPEG stream and status polling are built for a single
  local user (one webcam, one browser tab) - not designed to serve many
  simultaneous viewers.

## Future enhancements

Realistic next steps, clearly separate from what's implemented today:

- Face recognition (identifying *who* a detected face belongs to).
- Multi-face tracking across frames (currently each frame is detected
  independently, with no identity continuity between frames).
- Blink detection / drowsiness detection.
- Basic emotion detection.
- Swapping Haar Cascades for a deep-learning detector (MTCNN, MediaPipe,
  or YuNet) for better accuracy on angled faces and poor lighting.
- An attendance-marking system built on top of face recognition.
- Performance profiling and optimization for lower-end hardware.
