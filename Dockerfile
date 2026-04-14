FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for package management
RUN pip install uv

# Clone LTX-2 repo and install
RUN git clone https://github.com/Lightricks/LTX-2.git /app/ltx2
WORKDIR /app/ltx2
RUN uv sync --frozen || pip install -e packages/ltx-core -e packages/ltx-pipelines

WORKDIR /app

# Install additional dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=300 -r requirements.txt

COPY handler.py .

CMD ["python", "-u", "handler.py"]
