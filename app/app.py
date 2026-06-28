"""
PPE Compliance Detection - Streamlit prototype
Industry 5.0 worker safety monitoring (YOLOv8)

Run locally:
    pip install ultralytics streamlit opencv-python pillow
    streamlit run app/app.py

Put your trained best.pt next to this file (or set its path in the sidebar).
"""
from pathlib import Path

import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="PPE Compliance - Industry 5.0",
                   page_icon="🦺", layout="wide")

# ----------------------------- Sidebar -----------------------------
st.sidebar.title("Settings")
model_path = st.sidebar.text_input("Model weights (best.pt)", value="best.pt")
conf = st.sidebar.slider("Confidence threshold", 0.10, 0.90, 0.30, 0.05)
st.sidebar.caption(
    "Lower confidence catches more violations (higher recall). "
    "0.30 is a sensible safety default - a false alarm beats a missed violation."
)

@st.cache_resource(show_spinner="Loading model...")
def load_model(path):
    return YOLO(path)

# --------------------------- Compliance ----------------------------
def summarize(result, names):
    """Return per-class counts and the subset that are PPE violations (NO-*)."""
    counts = {}
    for c in result.boxes.cls.tolist():
        name = names[int(c)]
        counts[name] = counts.get(name, 0) + 1
    violations = {k: v for k, v in counts.items() if k.upper().startswith("NO-")}
    return counts, violations

def show_compliance(result, names):
    counts, violations = summarize(result, names)
    people = counts.get("Person", 0)
    total = sum(counts.values())

    c1, c2, c3 = st.columns(3)
    c1.metric("People detected", people)
    c2.metric("Total detections", total)
    c3.metric("PPE violations", sum(violations.values()))

    if violations:
        st.error("NON-COMPLIANT - missing PPE detected: "
                 + ", ".join(f"{k} ({v})" for k, v in violations.items()))
    else:
        st.success("COMPLIANT - no missing-PPE detections")

    if counts:
        st.caption("Detection breakdown")
        st.bar_chart({k: [v] for k, v in sorted(counts.items())})

def detect_and_show(image, model):
    """Run inference on a PIL image and render annotated result + compliance."""
    result = model.predict(image, conf=conf, verbose=False)[0]
    st.image(result.plot()[:, :, ::-1], caption="Detections",
             use_container_width=True)  # plot() is BGR -> flip to RGB
    show_compliance(result, model.names)

# ------------------------------ Main -------------------------------
st.title("PPE Compliance Detection - Industry 5.0")
st.write(
    "Detects workers and their protective equipment, and flags anyone missing "
    "a hard hat, mask, or safety vest near machinery. Built on a YOLOv8 model "
    "fine-tuned on the Construction Site Safety dataset."
)

if not Path(model_path).exists():
    st.warning(
        f"Model file '{model_path}' not found. Download best.pt from your "
        "Google Drive training run and place it next to app.py, or set the "
        "path in the sidebar."
    )
    st.stop()

model = load_model(model_path)

tab_img, tab_snap, tab_live = st.tabs(
    ["Image upload", "Webcam snapshot", "Live webcam"]
)

with tab_img:
    up = st.file_uploader("Upload a site image", type=["jpg", "jpeg", "png"])
    if up:
        detect_and_show(Image.open(up).convert("RGB"), model)

with tab_snap:
    shot = st.camera_input("Take a photo")
    if shot:
        detect_and_show(Image.open(shot).convert("RGB"), model)

with tab_live:
    st.write("Live detection from your webcam (works when running locally).")
    st.caption("For the smoothest demo, the standalone webcam_demo.py is more fluid.")
    run = st.checkbox("Start camera")
    frame_slot = st.empty()
    if run:
        cap = cv2.VideoCapture(0)
        while run:
            ok, frame = cap.read()
            if not ok:
                st.error("Could not read from webcam.")
                break
            result = model.predict(frame, conf=conf, verbose=False)[0]
            frame_slot.image(result.plot(), channels="BGR",
                             use_container_width=True)
        cap.release()
