FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone LTX-2 repo and install packages
RUN git clone https://github.com/Lightricks/LTX-2.git /app/ltx2
RUN cd /app/ltx2 && pip install --no-cache-dir --timeout=300 -e packages/ltx-core && pip install --no-cache-dir --timeout=300 -e packages/ltx-pipelines
ENV PYTHONPATH="/app/ltx2/packages/ltx-core/src:/app/ltx2/packages/ltx-pipelines/src:${PYTHONPATH}"

# Install additional dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=300 -r requirements.txt

COPY handler.py .

CMD ["python", "-u", "handler.py"]
