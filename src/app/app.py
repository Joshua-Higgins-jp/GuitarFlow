from datetime import datetime
from pathlib import Path
from time import monotonic as time_monotonic
from typing import Optional
from uuid import uuid4

import streamlit as st
from PIL import Image
from PIL.ImageFile import ImageFile
from streamlit.runtime.uploaded_file_manager import UploadedFile

from schemas.inference_event import ClassProbabilities, InferenceEvent
from app_settings import APP_DEBUG_MODE
from config.globals import ClassLabels
from config.paths import MODELS_DIR
from models.prediction import classification_predict, load_classification_model
from monitoring.app_monitoring_datadog import send_inference_event
from utils.dt_timestamps import get_dt_now_utc
from utils.image_metadata import ImageMetadata


@st.cache_resource
def get_model(model_name: str = "guitar_classifier.pth"):
    model_path: Path = MODELS_DIR / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model {model_name} not found.")

    return load_classification_model(weights_path=model_path)


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GuitarFlow",
    page_icon="🎸",
    layout="centered",
)
if "session_id" not in st.session_state:
    st.session_state["session_id"] = uuid4()

st.title("🎸 GuitarFlow")
st.caption("Electric or Acoustic? Let the model decide.")

# ── Image input ────────────────────────────────────────────────────────────────
tab_upload, tab_camera = st.tabs(
    tabs=[
        "📁 Upload Photo",
        "📷 Take Photo"
    ]
)

image: Optional[Image.Image] = None
metadata: Optional[ImageMetadata] = None

with tab_upload:
    uploaded: Optional[UploadedFile] = st.file_uploader(
        label="Upload a guitar photo",
        type=["jpg", "jpeg", "png", "webp"],
        max_upload_size=5
    )
    if uploaded:
        image_uploaded_timestamp: datetime = get_dt_now_utc()
        metadata = ImageMetadata.from_uploaded(uploaded=uploaded)
        image = Image.open(fp=uploaded)


with tab_camera:
    captured = st.camera_input("Take a photo of your guitar")
    if captured:
        image_uploaded_timestamp: datetime = get_dt_now_utc()
        image: Optional[ImageFile] = Image.open(fp=captured)
        if image:
            metadata = ImageMetadata.from_pil(image=image)


# ── Inference ──────────────────────────────────────────────────────────────────
if image and metadata:
    if metadata.is_valid():
        st.image(image, caption="Your guitar", width="stretch")

        # then in your inference block:
        with st.spinner("Classifying..."):
            model = get_model()
            inference_start_time: float = time_monotonic()
            scores = classification_predict(image, model)
            inference_end_time: float = time_monotonic()
            inference_time_ms: float = 1000 * round(number=(inference_end_time - inference_start_time), ndigits=4)

            predicted = max(scores, key=scores.__getitem__)
            confidence = scores[predicted] * 100

            event = InferenceEvent(
                image_capture_timestamp=image_uploaded_timestamp,
                session_id=st.session_state["session_id"],
                model_version="beta",
                image_filename=metadata.filename,
                image_hash=metadata.image_hash,
                image_width_px=metadata.width_px,
                image_height_px=metadata.height_px,
                image_format=metadata.image_format,
                image_file_size_bytes=metadata.file_size_bytes,
                num_channels=metadata.num_channels,
                predicted_class=ClassLabels(predicted),
                confidence_score=scores[predicted],
                all_class_probabilities=ClassProbabilities(**scores),
                inference_latency_ms=inference_time_ms,
            )

            send_inference_event(event=event)

            if APP_DEBUG_MODE:
                with st.expander("🔍 Debug — Inference Event", expanded=False):
                    st.json(event.model_dump(mode="json"))
                    st.bar_chart(scores)  # raw scores dict maps directly

    # ── Results display ────────────────────────────────────────────────────────
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Detection", value=predicted.capitalize())

    with col2:
        st.metric(label="Confidence", value=f"{confidence:.1f}%")
