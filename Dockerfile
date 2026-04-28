# ── Base ──────────────────────────────────────────────────────────────────────
FROM python:3.12-slim

# ── System deps ───────────────────────────────────────────────────────────────
# curl: healthcheck
# build-essential: some wheels need a C compiler
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user ─────────────────────────────────────────────────────────────
RUN useradd --create-home --shell /bin/bash appuser

# ── App directory ─────────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first so Docker cache layer is reused if deps don't change
COPY requirements.deploy.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.deploy.txt

# ── App source ────────────────────────────────────────────────────────────────
COPY src/ ./src/

# ── Model weights ─────────────────────────────────────────────────────────────
# Copy weights into the image for self-contained deploy.
# Alternative: download from HF Hub at startup (keeps image smaller).
#COPY src/guitar_classifier.pth ./src/guitar_classifier.pth

# ── Permissions ───────────────────────────────────────────────────────────────
RUN chown -R appuser:appuser /app

USER appuser

# ── Streamlit config ──────────────────────────────────────────────────────────
# HF Spaces expects the app on port 7860
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
# Disable the "Deploy to Streamlit Cloud" button — not relevant here
ENV STREAMLIT_SERVER_ENABLE_STATIC_SERVING=false
ENV PYTHONPATH=/app

EXPOSE 7860

# ── Healthcheck ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

ENV APP_ENV="PROD"

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["python", "-m", "streamlit", "run", "src/app/streamlit_app.py"]