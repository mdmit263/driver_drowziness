import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist

# -----------------------------

# Load alert sound (browser-based)

# -----------------------------

ALERT_WAV = "alert.wav"

def load_alert_sound():
try:
with open(ALERT_WAV, "rb") as f:
return f.read()
except:
return None

alert_sound_bytes = load_alert_sound()

# -----------------------------

# Eye Aspect Ratio (EAR)

# -----------------------------

def eye_aspect_ratio(eye):
A = dist.euclidean(eye[1], eye[5])
B = dist.euclidean(eye[2], eye[4])
C = dist.euclidean(eye[0], eye[3])
ear = (A + B) / (2.0 * C)
return ear

# -----------------------------

# MediaPipe Face Mesh

# -----------------------------

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

EAR_THRESH = 0.27
EAR_CONSEC_FRAMES = 12

# -----------------------------

# Video Processor

# -----------------------------

class VideoProcessor(VideoProcessorBase):
def **init**(self):
self.counter = 0
self.drowsy = False
self.face_mesh = mp_face_mesh.FaceMesh(
max_num_faces=1,
refine_landmarks=True,
min_detection_confidence=0.5,
min_tracking_confidence=0.5
)

```
def recv(self, frame):
    frm = frame.to_ndarray(format="bgr24")
    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
    result = self.face_mesh.process(rgb)

    if result.multi_face_landmarks:
        mesh = result.multi_face_landmarks[0]

        h, w, _ = frm.shape
        left_eye = [(int(mesh.landmark[i].x * w), int(mesh.landmark[i].y * h)) for i in LEFT_EYE_IDX]
        right_eye = [(int(mesh.landmark[i].x * w), int(mesh.landmark[i].y * h)) for i in RIGHT_EYE_IDX]

        ear_left = eye_aspect_ratio(left_eye)
        ear_right = eye_aspect_ratio(right_eye)
        ear = (ear_left + ear_right) / 2.0

        if ear < EAR_THRESH:
            self.counter += 1
            if self.counter >= EAR_CONSEC_FRAMES:
                self.drowsy = True
        else:
            self.counter = 0
            self.drowsy = False

        cv2.putText(frm, f"EAR: {ear:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if self.drowsy:
            cv2.putText(frm, "DROWSINESS ALERT!", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    return frame.from_ndarray(frm, format="bgr24")
```

# -----------------------------

# Streamlit UI

# -----------------------------

st.title("Driver Drowsiness Monitoring System")
st.write("Real-time EAR-based eye-closure drowsiness detection.")

if alert_sound_bytes:
if "should_play_alert" not in st.session_state:
st.session_state["should_play_alert"] = False

def webrtc_callback(frame):
    if vp.drowsy and alert_sound_bytes:
        st.session_state["should_play_alert"] = True

RTC_config = RTCConfiguration(
{"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

ctx = webrtc_streamer(
key="drowsy",
mode="recvonly",
rtc_configuration=RTC_config,
media_stream_constraints={"video": True, "audio": False},
video_processor_factory=VideoProcessor,
)

if ctx.video_processor:
vp: VideoProcessor = ctx.video_processor
if vp.drowsy and alert_sound_bytes:
st.session_state["should_play_alert"] = True

if st.session_state.get("should_play_alert", False):
st.audio(alert_sound_bytes, autoplay=True)
st.session_state["should_play_alert"] = False

