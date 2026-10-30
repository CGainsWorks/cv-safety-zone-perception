# Runtime image.  CPU, ONNX Runtime, no torch and no ultralytics.
#
# The AGPL-3.0 export toolchain deliberately does not appear here; see
# docs/licences.md and docker/Dockerfile.export.  The result is an image whose
# code dependencies are permissively licensed (python-can is LGPL-3.0); the
# AGPL-3.0 weights are mounted at run time and keep their licence.
#
#   docker build -f docker/Dockerfile -t szp:latest .
#   docker run --rm szp:latest                         # the demo
#   docker run --rm szp:latest python -m pytest tests/ -q
#
# The model weights are not baked in.  They are AGPL-derived and machine
# independent, so they are mounted or fetched at run time; without them
# everything except the real detector still works.

FROM python:3.12-slim-bookworm

# libGL and libglib are what opencv-python-headless still wants for its image
# codecs.  iproute2 and can-utils are for vcan: candump is the quickest way to
# see that the e-stop frames are real.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        iproute2 \
        can-utils \
        kmod \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY analysis/ ./analysis/
COPY tests/ ./tests/
COPY config/ ./config/
COPY data/ ./data/
COPY docs/ ./docs/
COPY pytest.ini .flake8 Makefile LICENSE README.md ./

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    SZP_MODEL=/app/models/yolov8n.onnx \
    SZP_DATA_ROOT=/data

# vcan needs a kernel module and NET_ADMIN, which a default container does not
# have.  Without it the pipeline uses python-can's in-process virtual bus and
# says so in the result file.  docker/docker-compose.yml grants the capability.
CMD ["python", "scripts/run_demo.py", "--out", "/app/results/demo.json"]
