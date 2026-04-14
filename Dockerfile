FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Clone LTX-2 repo and install with exact pinned versions
RUN git clone https://github.com/Lightricks/LTX-2.git /app/ltx2
WORKDIR /app/ltx2
RUN uv sync --frozen --no-dev
WORKDIR /app

# Install runpod and other deps into the uv venv
RUN /app/ltx2/.venv/bin/pip install runpod==1.7.9 huggingface_hub>=0.27.0 imageio[ffmpeg] requests Pillow>=10.0.0

COPY handler.py .

CMD ["/app/ltx2/.venv/bin/python", "-u", "handler.py"]
