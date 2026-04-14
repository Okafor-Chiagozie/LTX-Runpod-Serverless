# LTX-Video Serverless on RunPod — Setup Guide

## Overview
RunPod serverless endpoints for AI video generation using LTX-Video models. Three model tiers available.

## Model Tiers

| Model | Branch | Parameters | GPU | Quality | Cost/sec |
|---|---|---|---|---|---|
| **2B** | `2b-model` | 2B | 24 GB PRO | Good | $0.00031/s |
| **13B 0.9.8** | `main` | 13B | 80-96 GB | Better | $0.00076-0.00111/s |
| **LTX-2.3** | `ltx-2.3` | 22B | 80+ GB | Best | TBD |

## Architecture
- **GitHub Repo**: `Okafor-Chiagozie/LTX-Runpod-Serverless`
- **Deploy Method**: RunPod deploys directly from GitHub (no Docker Hub needed)
- **Model Storage**: Downloaded from HuggingFace on cold start (no network volume needed)
- **Benefit**: GPUs available from ALL datacenters, not locked to one region

## Files
- `handler.py` — RunPod serverless handler
- `requirements.txt` — Python dependencies
- `Dockerfile` — Container build config (PyTorch 2.8.0, CUDA 12.8.1)
- `test_endpoint.py` — Local script to test endpoint and download video
- `.env` — Local API key (not committed to git)

---

## Setup Steps

### 1. Create GitHub Repo
Push code to GitHub. RunPod builds the Docker image automatically.

### 2. Create RunPod Serverless Endpoint
1. Go to RunPod > **Serverless** > **Deploy** > **Custom deployment** > **Deploy from GitHub**
2. Connect GitHub account, select repo and branch
3. Configure endpoint (see settings below)
4. **Create endpoint** — RunPod builds automatically

### 3. Configure Endpoint Settings

#### 2B Model (`2b-model` branch)
- **GPU**: 24 GB PRO
- **Container disk**: 20 GB
- **Max workers**: 3
- **Active workers**: 0
- **Idle timeout**: 120 sec
- **Execution timeout**: 900 sec
- **No network volume**

#### 13B Model (`main` branch)
- **GPU**: 80 GB + 96 GB (fallback)
- **Container disk**: 60 GB
- **Max workers**: 3
- **Active workers**: 0
- **Idle timeout**: 120 sec
- **Execution timeout**: 1800 sec
- **No network volume**

### 4. Test Locally
Create `.env` file:
```
RUNPOD_API_KEY=your_api_key_here
```

Install dependencies:
```bash
pip install requests python-dotenv
```

Run test:
```bash
python test_endpoint.py
```

Video saves as `output.mp4`.

---

## API Usage

### 2B Model — Text-to-Video
```json
{
  "input": {
    "prompt": "A woman smiling in golden sunset light, cinematic",
    "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
    "num_frames": 49,
    "fps": 24,
    "width": 768,
    "height": 512,
    "num_inference_steps": 50,
    "decode_timestep": 0.03,
    "decode_noise_scale": 0.025
  }
}
```

### 2B Model — Image-to-Video
```json
{
  "input": {
    "image": "https://example.com/image.jpg",
    "prompt": "The scene comes to life with gentle motion",
    "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
    "num_frames": 49,
    "fps": 24,
    "width": 768,
    "height": 512,
    "num_inference_steps": 50,
    "decode_timestep": 0.03,
    "decode_noise_scale": 0.025
  }
}
```

### 13B Model — Text-to-Video (with upscaling)
```json
{
  "input": {
    "prompt": "A woman smiling in golden sunset light, cinematic",
    "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
    "num_frames": 97,
    "fps": 24,
    "width": 832,
    "height": 480,
    "num_inference_steps": 30,
    "guidance_scale": 5.0,
    "upscale": true,
    "denoise_strength": 0.4
  }
}
```

### 13B Model — Quick Test (no upscaling)
```json
{
  "input": {
    "prompt": "A woman smiling in golden sunset light, cinematic",
    "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
    "num_frames": 49,
    "fps": 24,
    "width": 832,
    "height": 480,
    "num_inference_steps": 30,
    "upscale": false
  }
}
```

## Input Parameters

### 2B Model
| Parameter | Default | Description |
|---|---|---|
| `prompt` | `""` | Text description of desired video |
| `image` | `null` | Image URL or base64 for image-to-video |
| `negative_prompt` | `"worst quality, inconsistent motion, blurry, jittery, distorted"` | What to avoid |
| `num_frames` | `49` | Number of frames (~2 sec at 24fps) |
| `fps` | `24` | Frames per second |
| `width` | `768` | Video width (divisible by 32) |
| `height` | `512` | Video height (divisible by 32) |
| `num_inference_steps` | `50` | Denoising steps |
| `guidance_scale` | `3.0` | Prompt adherence |
| `decode_timestep` | `0.03` | VAE decode timestep (critical for quality) |
| `decode_noise_scale` | `0.025` | VAE decode noise (critical for quality) |
| `seed` | `-1` | Random seed (-1 = random) |

### 13B Model
| Parameter | Default | Description |
|---|---|---|
| `prompt` | `""` | Text description |
| `image` | `null` | Image for image-to-video |
| `negative_prompt` | `"worst quality, inconsistent motion, blurry, jittery, distorted"` | What to avoid |
| `num_frames` | `97` | Number of frames (~4 sec at 24fps) |
| `fps` | `24` | Frames per second |
| `width` | `832` | Output width (divisible by 32) |
| `height` | `480` | Output height (divisible by 32) |
| `num_inference_steps` | `30` | Denoising steps |
| `guidance_scale` | `5.0` | Prompt adherence |
| `upscale` | `true` | Enable 3-step upscale pipeline |
| `denoise_strength` | `0.4` | Upscale denoising strength |
| `seed` | `-1` | Random seed (-1 = random) |

## Cold Start Times
Models download from HuggingFace on each cold start:
- **2B**: ~3-4 min (download + load)
- **13B**: ~10-15 min (download + load)

After cold start, worker stays warm for `idle_timeout` seconds. Batch your clips to avoid repeated cold starts.

## Cost Estimates
- **2B — 35 clips**: ~$0.50-1.00 + one cold start (~$0.06)
- **13B — 35 clips**: ~$2-4.00 + one cold start (~$0.50)
- **No storage costs** (no network volume)

## Key Lessons / Troubleshooting

### Quality is garbled/noisy (2B model)
You MUST include `decode_timestep=0.03` and `decode_noise_scale=0.025`. Without these, the VAE produces garbage.

### CUDA out of memory on 24GB (2B model)
Use `enable_sequential_cpu_offload()`. Max recommended resolution: 768x512.

### Resolution must be divisible by 32
Both width and height must be divisible by 32 (e.g., 768x512, 832x480).

### Network volume locks you to one datacenter
Don't use network volumes — download from HuggingFace instead. This gives access to GPUs in ALL datacenters.

### RTX 5090 not compatible with PyTorch 2.4.0
Use base image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (PyTorch 2.8.0).

## Docker Image Versions (legacy, before GitHub deploy)
| Version | Notes |
|---|---|
| v1-v4 | PyTorch 2.4.0, various errors |
| v5-v8 | Missing decode params, bad quality |
| v9 | PyTorch 2.8.0, correct params (replaced by GitHub deploy) |
