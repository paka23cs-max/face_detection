// Controls the Start/Stop/Reset buttons and keeps the on-page stats
// (status, face count, eye count) in sync with the Flask backend.
//
// The backend's /status response is the single source of truth for
// whether detection is active - not just the button clicks - because
// the backend can also stop on its own (e.g. the webcam disconnects).
// Polling continuously and always trusting the latest /status response
// is what keeps the buttons and video area from getting stuck out of
// sync with reality.

const videoStream = document.getElementById("video-stream");
const videoPlaceholder = document.getElementById("video-placeholder");
const statusValue = document.getElementById("status-value");
const faceCount = document.getElementById("face-count");
const eyeCount = document.getElementById("eye-count");
const startBtn = document.getElementById("start-btn");
const stopBtn = document.getElementById("stop-btn");
const resetBtn = document.getElementById("reset-btn");

let streaming = false;

async function postJSON(url) {
  const response = await fetch(url, { method: "POST" });
  return response.json();
}

function applyStatus(data) {
  faceCount.textContent = data.faces;
  eyeCount.textContent = data.eyes;

  if (data.active) {
    statusValue.textContent = "Active";
    statusValue.classList.remove("status-stopped");
    statusValue.classList.add("status-active");
  } else {
    statusValue.textContent = "Stopped";
    statusValue.classList.remove("status-active");
    statusValue.classList.add("status-stopped");
  }

  startBtn.disabled = data.active;
  stopBtn.disabled = !data.active;

  if (data.active && !streaming) {
    // Cache-bust so the browser opens a fresh /video_feed connection
    // instead of reusing a stale/closed one from a previous Start click.
    videoStream.src = "/video_feed?t=" + Date.now();
    videoStream.style.display = "block";
    videoPlaceholder.style.display = "none";
    streaming = true;
  } else if (!data.active && streaming) {
    videoStream.removeAttribute("src");
    videoStream.style.display = "none";
    videoPlaceholder.style.display = "flex";
    streaming = false;
  }
}

async function pollStatus() {
  try {
    const data = await fetch("/status").then((r) => r.json());
    applyStatus(data);
  } catch (err) {
    console.error("Could not reach /status:", err);
  }
}

startBtn.addEventListener("click", async () => {
  const data = await postJSON("/start");
  await pollStatus();
});

stopBtn.addEventListener("click", async () => {
  await postJSON("/stop");
  await pollStatus();
});

resetBtn.addEventListener("click", async () => {
  await postJSON("/stop");
  await postJSON("/reset");
  await pollStatus();
});

// Poll continuously (not just while we think detection is active) so the
// UI notices a backend-initiated stop, e.g. the webcam disconnecting.
pollStatus();
setInterval(pollStatus, 1000);
