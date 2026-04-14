# LTX-Video 2B Serverless on RunPod — Setup Guide

## Overview
RunPod serverless endpoint running LTX-Video 2B for text-to-video and image-to-video generation.

## Architecture
- **Model**: LTX-Video 2B (from `Lightricks/LTX-Video` diffusers pipeline)
- **Runtime**: RunPod Serverless
- **Docker Image**: `collincity/ltx-serverless:v9`
- **Base Image**: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (PyTorch 2.8.0, CUDA 12.8.1)
- **Network Volume**: `LTX_Production_Data` (50GB, EU-RO-1)
- **Model Path**: `/runpod-volume/ltx-model/`
- **GPU**: 24 GB PRO (RTX 4090) with sequential CPU offloading
- **Endpoint ID**: `swrgif95vdcviz`

## Files
- `handler.py` — RunPod serverless handler (text-to-video + image-to-video)
- `requirements.txt` — Python dependencies
- `Dockerfile` — Container build config
- `test_endpoint.py` — Local script to test the endpoint and download video

---

## Step-by-Step Setup

### 1. Download Model to RunPod Network Volume

1. Create a **50GB Network Volume** on RunPod (region: EU-RO-1)
2. Deploy a temporary **GPU Pod** (cheapest available, e.g. RTX 2000 Ada)
   - Template: **RunPod Pytorch 2.8.0**
   - Attach the network volume
3. Open **Jupyter Lab** terminal and run:

```bash
pip install huggingface_hub hf-xet
```

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Lightricks/LTX-Video', local_dir='/workspace/ltx-model', allow_patterns=['model_index.json', 'scheduler/*', 'text_encoder/*', 'tokenizer/*', 'transformer/*', 'vae/*'])"
```

4. **Terminate the pod** immediately after download completes (model stays on network volume)

> **Note**: Only download essential files (~15GB), NOT the full repo (111GB). Use `allow_patterns` to filter.

### 2. Build Docker Image (Local Machine)

```bash
docker build -t collincity/ltx-serverless:v9 .
```

Build takes ~30-60 minutes on first run (downloads base image + pip packages).

### 3. Push to Docker Hub

```bash
docker login
docker push collincity/ltx-serverless:v9
```

### 4. Create RunPod Serverless Endpoint

1. Go to RunPod > **Serverless** > **Get Started** > **Custom deployment** > **Deploy from Docker registry**
2. Set container image: `collincity/ltx-serverless:v9`
3. Click **Create endpoint**
4. Configure:
   - **GPU**: 24 GB PRO ($0.00031/s)
   - **Max Workers**: 3
   - **Active Workers**: 0
   - **Idle Timeout**: 120 sec
   - **Execution Timeout**: 900 sec
5. Attach network volume: **Manage** > **Edit Endpoint** > **Advanced** > **Network Volumes** > select `LTX_Production_Data`
6. Save

---

## API Usage

### Text-to-Video
```json
{
  "input": {
    "prompt": "A cat walking through a sunny garden with flowers, cinematic lighting, high quality",
    "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
    "num_frames": 161,
    "fps": 24,
    "width": 1024,
    "height": 576,
    "num_inference_steps": 50,
    "decode_timestep": 0.03,
    "decode_noise_scale": 0.025
  }
}
```

### Image-to-Video
```json
{
  "input": {
    "image": "https://example.com/image.jpg",
    "prompt": "The scene comes to life with gentle motion",
    "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
    "num_frames": 161,
    "fps": 24,
    "width": 1024,
    "height": 576,
    "num_inference_steps": 50,
    "decode_timestep": 0.03,
    "decode_noise_scale": 0.025
  }
}
```

### Image + Text to Video
```json
{
  "input": {
    "image": "<base64-encoded-image-or-url>",
    "prompt": "A description of the desired video",
    "num_frames": 161,
    "fps": 24,
    "width": 1024,
    "height": 576,
    "num_inference_steps": 50,
    "decode_timestep": 0.03,
    "decode_noise_scale": 0.025
  }
}
```

### Response
```json
{
  "video_base64": "<base64-encoded-mp4>",
  "format": "mp4",
  "frames": 161,
  "fps": 24,
  "width": 1024,
  "height": 576
}
```

## Input Parameters

| Parameter | Default | Description |
|---|---|---|
| `prompt` | `""` | Text description of desired video |
| `image` | `null` | Image URL or base64 for image-to-video |
| `negative_prompt` | `"worst quality, inconsistent motion, blurry, jittery, distorted"` | What to avoid |
| `num_frames` | `161` | Number of frames (~6.7 sec at 24fps) |
| `fps` | `24` | Frames per second |
| `width` | `1024` | Video width (must be divisible by 32) |
| `height` | `576` | Video height (must be divisible by 32) |
| `num_inference_steps` | `50` | Denoising steps (higher = better quality, slower) |
| `guidance_scale` | `3.0` | How closely to follow the prompt |
| `decode_timestep` | `0.03` | VAE decode timestep (important for quality) |
| `decode_noise_scale` | `0.025` | VAE decode noise scale (important for quality) |
| `seed` | `-1` | Random seed (-1 = random) |

## Testing Locally

Use `test_endpoint.py` to send a request and save the video as MP4:

```bash
python test_endpoint.py
```

Edit `API_KEY` and `ENDPOINT_ID` in the script before running.

## Cost Estimates
- **35 clips of ~7 seconds**: ~$0.50-1.00 (2B model on 24GB PRO)
- **Network volume**: $0.07/GB/month = ~$3.50/month for 50GB
- **Idle cost**: $0.00 (scales to zero)

## Key Lessons / Troubleshooting

### Quality is garbled/noisy
You MUST include `decode_timestep=0.03` and `decode_noise_scale=0.025` in every request. Without these, the VAE produces garbage output.

### CUDA out of memory on 24GB
Use `enable_sequential_cpu_offload()` instead of `.to("cuda")`. This moves each layer to GPU one at a time during inference.

### RTX 5090 not compatible
PyTorch 2.4.0 doesn't support RTX 5090 (sm_120). Use base image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` with PyTorch 2.8.0.

### diffusers import error with torch 2.4.0
Latest diffusers is incompatible with PyTorch 2.4.0. Either pin `diffusers==0.32.2` or upgrade to PyTorch 2.8.0 base image.

### Timeout downloading torch during build
Add `--timeout=300` to pip install in Dockerfile.

### Disk quota exceeded on network volume
Check usage: `df -h /workspace && du -sh /workspace/*/`
Delete unused folders to free space.

### Model download shows 111GB
You're downloading the entire repo. Use `allow_patterns` to filter to only essential files.

### Network volume not showing in serverless config
Create the endpoint first, then attach the volume via **Manage** > **Edit Endpoint** > **Advanced** > **Network Volumes**.

## Docker Image Versions
| Version | Status | Notes |
|---|---|---|
| v1-v4 | Deprecated | PyTorch 2.4.0, various import/OOM errors |
| v5-v8 | Deprecated | Missing decode params, bad quality |
| **v9** | **Current** | PyTorch 2.8.0, latest diffusers, correct decode params |
