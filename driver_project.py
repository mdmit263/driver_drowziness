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
import base64

# ---------------------- Configuration ----------------------
ALERT_WAV = "alert.wav"  # must be in repo root
TWILIO_SID = "YOUR_ACCOUNT_SID"
TWILIO_TOKEN = "YOUR_AUTH_TOKEN"
TWILIO_FROM = "+1234567890"
TWILIO_TO = "+19876543210"

# Detection tuning
EAR_THRESHOLD = 0.23
EAR_CONSEC_FRAMES = 30  # eyes must be closed ~1 sec
MAR_THRESHOLD = 0.65
NOD_THRESHOLD = 20
BLINK_MAX_FRAMES = 4
MOVING_AVG_FRAMES = 3

# UI
UI_REFRESH_SEC = 0.5

# Twilio cooldown (per message text)
SMS_COOLDOWN = 120

# queue where processor will push events (thread-safe)
EVENT_QUEUE: Queue = Queue()

# cooldown tracker for SMS
_sms_last_sent = defaultdict(lambda: 0.0)

# ---------------- Twilio client ----------------
try:
    twilio_client = Client(TWILIO_SID, TWILIO_TOKEN)
except Exception:
    twilio_client = None

# ---------------- Mediapipe FaceMesh ----------------
mp_face_mesh = face_mesh
face_mesh_model = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
OUTER_LIPS = [61, 291, 0, 17, 13, 14, 312, 308]

# ---------------- Streamlit session ----------------
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
if "alert_sound" not in st.session_state:
    try:
        with open(ALERT_WAV, "rb") as f:
            st.session_state["alert_sound"] = base64.b64encode(f.read()).decode()
    except Exception as e:
        st.warning("Could not load alert.wav: " + str(e))
if "alert_played" not in st.session_state:
    st.session_state["alert_played"] = False

# ---------------- UI Layout ----------------
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

# ---------------- Helper Functions ----------------
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


# ---------------- Browser Alert Sound ----------------
def play_alert_sound():
    if not st.session_state["alert_played"] and "alert_sound" in st.session_state:
        audio_b64 = st.session_state["alert_sound"]
        st.markdown(f"""
            <audio autoplay loop>
                <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
            </audio>
        """, unsafe_allow_html=True)
        st.session_state["alert_played"] = True


def stop_alert_sound():
    st.session_state["alert_played"] = False


# ---------------- Video Processor ----------------
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.closed_counter = 0
        self.recent_ear = deque(maxlen=MOVING_AVG_FRAMES)
        self.recent_mar = deque(maxlen=5)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if img is None or img.size == 0:
            return av.VideoFrame.from_ndarray(np.zeros((480, 640, 3), np.uint8), format="bgr24")

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
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        EAR = MAR = pitch = 0.0
        event_emitted = False

        multi_face_landmarks = getattr(results, "multi_face_landmarks", None)
        if multi_face_landmarks:
            landmarks = multi_face_landmarks[0].landmark

            left_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE]
            right_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE]
            mouth_pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in OUTER_LIPS]

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            EAR = float((left_ear + right_ear) / 2.0)
            self.recent_ear.append(EAR)
            smooth_ear = float(np.mean(self.recent_ear))

            MAR = float(mouth_aspect_ratio(mouth_pts))
            self.recent_mar.append(MAR)

            pitch = float(head_pitch(landmarks, w, h))

            # Eye closure detection
            if smooth_ear < EAR_THRESHOLD:
                self.closed_counter += 1
            else:
                if 0 < self.closed_counter <= BLINK_MAX_FRAMES:
                    pass
                self.closed_counter = 0
                stop_alert_sound()

            # Drowsy
            if self.closed_counter >= EAR_CONSEC_FRAMES:
                EVENT_QUEUE.put({
                    "ts": time.strftime("%H:%M:%S"),
                    "event": "Drowsy",
                    "EAR": round(smooth_ear, 3),
                    "MAR": round(MAR, 3),
                    "pitch": round(pitch, 2),
                    "sms": "⚠️ Driver Drowsiness Detected!"
                })
                play_alert_sound()
                self.closed_counter = EAR_CONSEC_FRAMES // 2
                event_emitted = True

            # Yawn
            if MAR > MAR_THRESHOLD:
                EVENT_QUEUE.put({
                    "ts": time.strftime("%H:%M:%S"),
                    "event": "Yawn",
                    "EAR": round(smooth_ear, 3),
                    "MAR": round(MAR, 3),
                    "pitch": round(pitch, 2),
                    "sms": "⚠️ Driver Yawning Detected!"
                })

            # Head nod
            if abs(pitch) > NOD_THRESHOLD:
                EVENT_QUEUE.put({
                    "ts": time.strftime("%H:%M:%S"),
                    "event": "Head Nod",
                    "EAR": round(smooth_ear, 3),
                    "MAR": round(MAR, 3),
                    "pitch": round(pitch, 2),
                    "sms": "⚠️ Head Nodding Detected!"
                })

            # Draw landmarks
            for (x, y) in (left_eye + right_eye + mouth_pts):
                cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

            cv2.putText(img, f"EAR:{smooth_ear:.2f} MAR:{MAR:.2f} Pitch:{pitch:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 20), 2)
            if event_emitted:
                cv2.putText(img, "⚠️ DROWSINESS ALERT! ⚠️", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------- WebRTC ----------------
webrtc_ctx = webrtc_streamer(
    key="drowsiness",
    video_processor_factory=DrowsinessProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

# ---------------- Event Queue UI ----------------
def drain_event_queue_and_update():
    updated = False
    while True:
        try:
            item = EVENT_QUEUE.get_nowait()
        except Empty:
            break

        st.session_state.EVENTS_DF = pd.concat([
            st.session_state.EVENTS_DF,
            pd.DataFrame([[item["ts"], item["event"], item["EAR"], item["MAR"], item["pitch"]]],
                         columns=st.session_state.EVENTS_DF.columns)
        ], ignore_index=True)

        if len(st.session_state.EVENTS_DF) > 200:
            st.session_state.EVENTS_DF = st.session_state.EVENTS_DF.iloc[-200:].reset_index(drop=True)

        st.session_state.EAR_HISTORY.append(item["EAR"])
        st.session_state.MAR_HISTORY.append(item["MAR"])

        sms_text = item.get("sms")
        if sms_text:
            send_sms_background(sms_text)

        updated = True
    return updated


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


updated = drain_event_queue_and_update()
show_ui()

if manual_refresh:
    st.rerun()
if updated:
    st.rerun()
