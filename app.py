# app.py
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import plotly.express as px
import time

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "brain_tumor_model_vgg16.h5"
IMAGE_SIZE = 128
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

st.set_page_config(
    page_title="Brain Tumor Detector",
    page_icon="🧠",
    layout="centered"
)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("📘 Project Information")
st.sidebar.markdown("""
## 🧠 AI Brain Tumor Detection System (Complete Project Details)

**Project Overview**
- Deep learning-based MRI brain scan classification into 4 classes.
- Trained on thousands of augmented MRI images.

**Objectives**
- Fast, accurate classification
- Clean demo-ready UI
- Explainable outputs (probabilities)

**Classes**
- Glioma
- Meningioma
- Pituitary
- No Tumor

**Tech Stack**
- Streamlit, TensorFlow/Keras, OpenCV, NumPy, Plotly

**How it works**
1. Upload MRI image
2. Image preprocessing (resize & normalize)
3. Model predicts class probabilities
4. Display prediction + probability chart

**Disclaimer**
This is a demo tool for educational purposes — not a medical diagnosis system.
""")


# -----------------------------
# HEADER
# -----------------------------
st.title("🧠 AI Brain Tumor Detection")
st.caption("Upload MRI → Predict Tumor → Check Probability Graph")

st.divider()

# -----------------------------
# UPLOAD
# -----------------------------
st.subheader("📤 Step 1 — Upload MRI Image")
uploaded = st.file_uploader("Upload MRI (JPG/PNG)", type=["jpg", "jpeg", "png"])

# -----------------------------
# CACHED MODEL
# -----------------------------
@st.cache_resource
def load_model_cached(path):
    return tf.keras.models.load_model(path)

# -----------------------------
# PREPROCESS
# -----------------------------
def preprocess(img):
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

# -----------------------------
# MAIN LOGIC
# -----------------------------
if uploaded:
    # PREVIEW
    st.subheader("🖼️ Step 2 — Preview")
    image = Image.open(uploaded).convert("RGB")
    st.image(image, use_container_width=True)

    st.subheader("🤖 Step 3 — Predict")
    run = st.button("✨ Run Prediction")

    if run:
        # Progress bar
        progress = st.progress(0)
        for i in range(6):
            time.sleep(0.12)
            progress.progress(int((i + 1) * (100 / 6)))

        model = load_model_cached(MODEL_PATH)
        arr = preprocess(image)
        preds = model.predict(arr)

        idx = int(np.argmax(preds[0]))
        label = CLASS_NAMES[idx]
        conf = float(preds[0][idx])

        st.success(f"🎯 Prediction: **{label}**")
        st.info(f"Confidence: **{conf*100:.2f}%**")

        # GRAPH
        st.subheader("📊 Class Probabilities")
        prob_map = {cls: float(p) for cls, p in zip(CLASS_NAMES, preds[0])}

        fig = px.bar(
            x=list(prob_map.values()),
            y=list(prob_map.keys()),
            orientation="h",
            text=[f"{v*100:.2f}%" for v in prob_map.values()],
        )

        st.plotly_chart(fig, use_container_width=True)

        # Download
        result = {
            "prediction": label,
            "confidence": conf,
            "probabilities": prob_map
        }
        st.download_button("📄 Download Result (JSON)", data=str(result), file_name="prediction.json")
