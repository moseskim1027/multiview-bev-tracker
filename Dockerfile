FROM python:3.11-slim

WORKDIR /app

# System libraries required by OpenCV and Torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifest first for better layer caching
COPY pyproject.toml ./

# Install core dependencies
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install torchreid from source (not on PyPI)
RUN pip install --no-cache-dir \
    git+https://github.com/KaiyangZhou/deep-person-reid.git

# Install the project and remaining dependencies
COPY . .
RUN pip install --no-cache-dir -e ".[dev]"

# Weights and data are mounted at runtime (see docker-compose.yml)
VOLUME ["/app/data", "/app/models"]

CMD ["python", "-m", "bev_tracker.pipeline", "--config", "configs/default.yaml"]
