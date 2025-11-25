import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # quieter TF logs

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import pandas as pd
from mediapipe.python.solutions import face_mesh
import time
from twilio.rest import Client
import av
import threading
from queue import Queue, Empty
from collections import deque, defaultdict

# ---------------------- Configuration ----------------------
ALERT_WAV = r"alert.wav"   # <-- update this
TWILIO_SID = "YOUR_ACCOUNT_SID"
TWILIO_TOKEN = "YOUR_AUTH_TOKEN"
TWILIO_FROM = "+1234567890"
TWILIO_TO = "+19876543210"

# Detection tuning
EAR_THRESHOLD = 0.23
EAR_CONSEC_FRAMES = 30     # eyes must be closed for ~1 sec
MAR_THRESHOLD = 0.65       # strong yawn, not minor mouth open
NOD_THRESHOLD = 20         # clear downward head nod
BLINK_MAX_FRAMES = 4
MOVING_AVG_FRAMES = 3

# UI
UI_REFRESH_SEC = 0.5

# Twilio cooldown (per message text)
SMS_COOLDOWN = 120

# -------------------------------------------------------------------
# Twilio client (used by background thread) - keep even if not used
try:
    twilio_client = Client(TWILIO_SID, TWILIO_TOKEN)
except Exception:
    twilio_client = None

# queue where processor will push events (thread-safe)
EVENT_QUEUE: Queue = Queue()

# cooldown tracker for SMS
_sms_last_sent = defaultdict(lambda: 0.0)

# -------------------------------------------------------------------
# Mediapipe FaceMesh (correct API for 0.10.x)
mp_face_mesh = face_mesh
face_mesh_model = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# landmark index sets
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
OUTER_LIPS = [61, 291, 0, 17, 13, 14, 312, 308]

# ---------------- Streamlit session initialization ----------------
st.set_page_config(layout="wide")
st.title("🚗 Driver Drowsiness Detection")

if "EVENTS_DF" not in st.session_state:
    st.session_state["EVENTS_DF"] = pd.DataFrame(columns=["Timestamp", "Event", "EAR", "MAR", "Pitch"])
if "EAR_HISTORY" not in st.session_state:
    st.session_state["EAR_HISTORY"] = []
if "MAR_HISTORY" not in st.session_state:
    st.session_state["MAR_HISTORY"] = []
if "last_ui_update" not in st.session_state:
    st.session_state["last_ui_update"] = 0.0

col1, col2 = st.columns((2, 1))
with col1:
    st.write("Camera stream (allow camera in browser).")
with col2:
    st.checkbox("Night Mode / IR Support", key="NIGHT_MODE")
    st.write("Event log (most recent on top):")
    events_container = st.empty()
    st.write("EAR / MAR history:")
    chart_container = st.empty()
    st.write("Controls:")
    start_button = st.button("Start Stream", key="start_button")
    stop_button = st.button("Stop Stream", key="stop_button")
    manual_refresh = st.button("Refresh UI", key="manual_refresh")

# ---------------- Helper functions ----------------
def send_sms_background(message: str):
    now = time.time()
    last = _sms_last_sent[message]
    if now - last < SMS_COOLDOWN:
        return
    _sms_last_sent[message] = now

    def _send():
        if twilio_client is None:
            print("Twilio not configured - skipping SMS:", message)
            return
        try:
            twilio_client.messages.create(body=message, from_=TWILIO_FROM, to=TWILIO_TO)
        except Exception as e:
            print("Twilio send failed:", e)

    threading.Thread(target=_send, daemon=True).start()


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C) if C != 0 else 0.0


def mouth_aspect_ratio(mouth):
