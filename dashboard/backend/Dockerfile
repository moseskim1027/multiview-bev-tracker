# Build context must be the project root:
#   docker build -f dashboard/backend/Dockerfile .

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install project runtime dependencies ──────────────────────────────────────
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# torchreid build prerequisites
RUN pip install --no-cache-dir Cython tensorboard

# torchreid from source (not on PyPI)
RUN pip install --no-cache-dir --no-build-isolation \
        git+https://github.com/KaiyangZhou/deep-person-reid.git

# ── Install bev_tracker package ───────────────────────────────────────────────
COPY pyproject.toml ./
COPY src/ src/
RUN pip install --no-cache-dir -e .

# ── Install dashboard backend deps ────────────────────────────────────────────
COPY dashboard/backend/requirements.txt dashboard_requirements.txt
RUN pip install --no-cache-dir -r dashboard_requirements.txt

# ── Copy backend source ───────────────────────────────────────────────────────
COPY dashboard/backend/ dashboard/backend/

WORKDIR /app/dashboard/backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
