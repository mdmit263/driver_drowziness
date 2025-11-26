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
import pygame
import threading
from queue import Queue, Empty
from collections import deque, defaultdict





# ---------------------- Configuration ----------------------
ALERT_WAV = r"D:/downloads/alert.wav"   # <-- update this
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
# Init pygame (safe guard)
try:
    pygame.mixer.init()
    try:
        pygame.mixer.music.load(ALERT_WAV)
    except Exception as e:
        print("Warning: could not load alert.wav:", e)
except Exception as e:
    print("Warning: pygame init failed:", e)

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

# 2️⃣ Correct reference to FaceMesh
mp_face_mesh = face_mesh

# 3️⃣ Initialize the model
face_mesh_model = mp_face_mesh.FaceMesh(
    max_num_faces=1,        # correct param for 0.10.21
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
    """Send SMS in background with cooldown per message text."""
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
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)


def mouth_aspect_ratio(mouth):
    if len(mouth) < 7:
        return 0.0
    A = distance.euclidean(mouth[2], mouth[6])
    B = distance.euclidean(mouth[3], mouth[5])
    C = distance.euclidean(mouth[0], mouth[1])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)


def head_pitch(landmarks, w, h):
    nose = np.array([landmarks[1].x * w, landmarks[1].y * h])
    chin = np.array([landmarks[152].x * w, landmarks[152].y * h])
    dy = chin[1] - nose[1]
    dx = chin[0] - nose[0]
    return np.degrees(np.arctan2(dy, dx))


# ---------------- Video processor ----------------
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.closed_counter = 0
        self.recent_ear = deque(maxlen=MOVING_AVG_FRAMES)
        self.recent_mar = deque(maxlen=5)

    def recv(self, frame):
        # Convert to ndarray (bgr24) and ensure uint8
        img = frame.to_ndarray(format="bgr24")
        if img is None or img.size == 0:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            return av.VideoFrame.from_ndarray(blank, format="bgr24")

        img = img.astype(np.uint8)
        h, w = img.shape[:2]

        # Night mode
        if st.session_state.get("NIGHT_MODE", False):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            results = face_mesh_model.process(rgb)
        except Exception as e:
            print("FaceMesh processing error:", e)
            img = frame.to_ndarray(format="bgr24")
            return av.VideoFrame.from_ndarray(img, format="bgr24")


        event_emitted = False
        EAR = 0.0
        MAR = 0.0
        pitch = 0.0

        # safely get multi_face_landmarks
        multi_face_landmarks = getattr(results, "multi_face_landmarks", None)
        if multi_face_landmarks:
            landmarks = multi_face_landmarks[0].landmark

            left_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE]
            right_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE]

    # EAR / MAR / pitch
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            EAR = float((left_ear + right_ear) / 2.0)

            self.recent_ear.append(EAR)
            smooth_ear = float(np.mean(self.recent_ear))

            mouth_pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in OUTER_LIPS]
            MAR = float(mouth_aspect_ratio(mouth_pts))
            self.recent_mar.append(MAR)

            pitch = float(head_pitch(landmarks, w, h))



            # detection logic
            if smooth_ear < EAR_THRESHOLD:
                self.closed_counter += 1
            else:
                # ignore quick blinks
                if 0 < self.closed_counter <= BLINK_MAX_FRAMES:
                    pass
                self.closed_counter = 0

            # Drowsy
            if self.closed_counter >= EAR_CONSEC_FRAMES:
                # emit event once per detection window
                EVENT_QUEUE.put({
                    "ts": time.strftime("%H:%M:%S"),
                    "event": "Drowsy",
                    "EAR": round(smooth_ear, 3),
                    "MAR": round(MAR, 3),
                    "pitch": round(pitch, 2),
                    "sms": "⚠️ Driver Drowsiness Detected!"
                })
                event_emitted = True
                # quick siren control
                try:
                    if not pygame.mixer.music.get_busy():
                        pygame.mixer.music.play(-1)
                except Exception:
                    pass
                # reduce counter to avoid continuous flood
                self.closed_counter = EAR_CONSEC_FRAMES // 2
            else:
                # stop siren immediately when eyes open
                try:
                    if pygame.mixer.music.get_busy() and smooth_ear >= EAR_THRESHOLD:
                        pygame.mixer.music.stop()
                except Exception:
                    pass

            # yawn
            if MAR > MAR_THRESHOLD:
                EVENT_QUEUE.put({
                    "ts": time.strftime("%H:%M:%S"),
                    "event": "Yawn",
                    "EAR": round(smooth_ear, 3),
                    "MAR": round(MAR, 3),
                    "pitch": round(pitch, 2),
                    "sms": "⚠️ Driver Yawning Detected!"
                })

            # head nod
            if abs(pitch) > NOD_THRESHOLD:
                EVENT_QUEUE.put({
                    "ts": time.strftime("%H:%M:%S"),
                    "event": "Head Nod",
                    "EAR": round(smooth_ear, 3),
                    "MAR": round(MAR, 3),
                    "pitch": round(pitch, 2),
                    "sms": "⚠️ Head Nodding Detected!"
                })

            # draw landmarks
            for (x, y) in (left_eye + right_eye + mouth_pts):
                cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

            cv2.putText(img, f"EAR:{smooth_ear:.2f} MAR:{MAR:.2f} Pitch:{pitch:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 20), 2)

            if event_emitted and int(time.time() * 2) % 2 == 0:
                cv2.putText(img, "⚠️ DROWSINESS ALERT! ⚠️", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

        # ensure uint8
        img = img.astype(np.uint8)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------- Start WebRTC with STUN ----------------
webrtc_ctx = webrtc_streamer(
    key="drowsiness",
    video_processor_factory=DrowsinessProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)


# ---------------- UI update function (poll event queue) ----------------
def drain_event_queue_and_update():
    updated = False
    while True:
        try:
            item = EVENT_QUEUE.get_nowait()
        except Empty:
            break

        # append to session_state DataFrame
        st.session_state.EVENTS_DF = pd.concat([
            st.session_state.EVENTS_DF,
            pd.DataFrame([[item["ts"], item["event"], item["EAR"], item["MAR"], item["pitch"]]],
                         columns=st.session_state.EVENTS_DF.columns)
        ], ignore_index=True)

        # trim
        if len(st.session_state.EVENTS_DF) > 200:
            st.session_state.EVENTS_DF = st.session_state.EVENTS_DF.iloc[-200:].reset_index(drop=True)

        # append histories
        st.session_state.EAR_HISTORY.append(item["EAR"])
        st.session_state.MAR_HISTORY.append(item["MAR"])

        # send sms if provided (background)
        sms_text = item.get("sms")
        if sms_text:
            send_sms_background(sms_text)

        updated = True

    return updated

# SESSION STATE INITIALIZATION
if "EVENTS_DF" not in st.session_state:
    st.session_state.EVENTS_DF = pd.DataFrame(columns=["Timestamp", "Event", "EAR", "MAR", "Pitch"])

if "EAR" not in st.session_state:
    st.session_state.EAR = None

if "MAR" not in st.session_state:
    st.session_state.MAR = None

if "PITCH" not in st.session_state:
    st.session_state.PITCH = None


# Show current events & chart
def show_ui():
    if not st.session_state.EVENTS_DF.empty:
        df_rev = st.session_state.EVENTS_DF.iloc[::-1]
        events_container.dataframe(df_rev.reset_index(drop=True), use_container_width=True)
    else:
        events_container.info("No events yet")

    ear = st.session_state.EAR_HISTORY[-200:]
    mar = st.session_state.MAR_HISTORY[-200:]
    if ear and mar:
        L = max(len(ear), len(mar))
        ear_p = ear[-L:] if len(ear) >= L else ([None] * (L - len(ear)) + ear)
        mar_p = mar[-L:] if len(mar) >= L else ([None] * (L - len(mar)) + mar)
        df_chart = pd.DataFrame({"EAR": ear_p, "MAR": mar_p})
        chart_container.line_chart(df_chart)
    else:
        chart_container.text("Waiting for EAR/MAR data...")


# ---------------- Main: drain queue, update UI, rerun only when updated ----------------
updated = drain_event_queue_and_update()
show_ui()

# Manual refresh button
if manual_refresh:
    st.rerun()

# If we drained events and added something, trigger a rerun so UI updates quickly.
# This avoids continuous reruns when there is no new data.
if updated:
    st.rerun()
