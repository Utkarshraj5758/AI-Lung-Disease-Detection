# syntax=docker/dockerfile:1
# CPU-only image for the Streamlit app (default) or the FastAPI service.
#
#   docker build -t lungxray .                              # Streamlit on :8501
#   docker build --build-arg SERVICE=api -t lungxray-api .  # FastAPI on :8000
#   docker run -p 8501:8501 -v "$PWD/artifacts:/app/artifacts:ro" lungxray
#
# Weights are not baked in: mount a bundle at /app/artifacts/best.pt, or set
# LUNGXRAY_HF_REPO to a Hugging Face model repo. LUNGXRAY_SERVICE=api|app also works
# at run time (docker run -e LUNGXRAY_SERVICE=api -p 8000:8000 ...).

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# libglib is needed by opencv-python-headless.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch first: a large layer that rarely changes.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install ".[api]" \
    && python -c "import cv2, pydicom, streamlit, torch, uvicorn, lungxray"

COPY configs ./configs
COPY app ./app
COPY api ./api

RUN useradd --create-home --uid 1000 lungxray && mkdir -p /app/artifacts
USER lungxray

# Declared late so both services share every layer above.
ARG SERVICE=app
ENV LUNGXRAY_SERVICE=${SERVICE}

EXPOSE 8501 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 CMD \
    python -c "import os, urllib.request; api = os.environ['LUNGXRAY_SERVICE'] == 'api'; \
url = 'http://127.0.0.1:8000/health' if api else 'http://127.0.0.1:8501/_stcore/health'; \
urllib.request.urlopen(url, timeout=4)"

CMD ["sh", "-c", "if [ \"$LUNGXRAY_SERVICE\" = api ]; then exec uvicorn api.main:app --host 0.0.0.0 --port 8000; else exec streamlit run app/streamlit_app.py; fi"]
