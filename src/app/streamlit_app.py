from pathlib import Path

import streamlit as st
from PIL import Image

from globals import MODELS_DIR
from src.prediction import classification_predict, load_classification_model


@st.cache_resource
def get_model(model_name: str = "guitar_classifier.pth"):
    model_path: Path = MODELS_DIR / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model {model_name} not found.")

    return load_classification_model(weights_path=model_path)

# ── Logging setup ──────────────────────────────────────────────────────────────
# logger.remove()
# logger.add(sys.stdout, level="INFO")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GuitarFlow",
    page_icon="🎸",
    layout="centered",
)

st.title("🎸 GuitarFlow")
st.caption("Electric or Acoustic? Let the model decide.")

# ── Image input ────────────────────────────────────────────────────────────────
tab_upload, tab_camera = st.tabs(["📁 Upload Photo", "📷 Take Photo"])

image: Image.Image | None = None

with tab_upload:
    uploaded = st.file_uploader(
        label="Upload a guitar photo",
        type=["jpg", "jpeg", "png", "webp"],
    )
    if uploaded:
        image = Image.open(uploaded)

with tab_camera:
    captured = st.camera_input("Take a photo of your guitar")
    if captured:
        image = Image.open(captured)

# ── Inference ──────────────────────────────────────────────────────────────────
if image:
    st.image(image, caption="Your guitar", width=True)

    # then in your inference block:
    with st.spinner("Classifying..."):
        model = get_model()
        scores = classification_predict(image, model)

        predicted = max(scores, key=scores.__getitem__)
        confidence = scores[predicted] * 100

    # ── Results display ────────────────────────────────────────────────────────
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Detection", value=predicted.capitalize())

    with col2:
        st.metric(label="Confidence", value=f"{confidence:.1f}%")
