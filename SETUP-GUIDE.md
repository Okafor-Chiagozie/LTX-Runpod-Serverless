# LTX-Video Serverless on RunPod — Setup Guide

## Overview
RunPod serverless endpoints for AI video generation. Four model tiers available across different branches.

## Model Tiers

| Model | Branch | Parameters | GPU | Speed | Quality | Cost/sec |
|---|---|---|---|---|---|---|
| **LTX 2B** | `2b-model` | 2B | 24 GB PRO | Fastest | Good | $0.00031/s |
| **LTX 13B 0.9.8** | `main` | 13B | 80-96 GB | Fast | Better | $0.00076-0.00111/s |
| **Wan2.1 14B** | `wan2.1` | 14B | 80-96 GB | Slow | High realism | $0.00076-0.00111/s |
| **LTX-2.3 22B** | `ltx-2.3` | 22B | 80-96 GB | Fast + Quality | Best | $0.00076-0.00111/s |

## Architecture
- **GitHub Repo**: `Okafor-Chiagozie/LTX-Runpod-Serverless`
- **Deploy Method**: RunPod deploys directly from GitHub branches
- **Model Storage**: Downloaded from HuggingFace on cold start (no network volume)
- **Benefit**: GPUs available from ALL datacenters, not locked to one region
- **Base Image**: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (PyTorch 2.8.0, CUDA 12.8.1)

## Files
- `handler.py` — RunPod serverless handler (varies per branch)
- `requirements.txt` — Python dependencies
- `Dockerfile` — Container build config
- `test_endpoint.py` — Local script to test endpoint and download video
- `.env` — Local API key (not committed to git)
- `SETUP-GUIDE.md` — This file

---

## Quick Setup (Any Model)

### 1. Push Code to GitHub
Each model lives on its own branch. Push code and RunPod builds automatically.

### 2. Create RunPod Serverless Endpoint
1. Go to RunPod > **Serverless** > **Deploy** > **Custom deployment** > **Deploy from GitHub**
2. Connect GitHub account, select repo and **branch**
3. Configure endpoint settings (see below)
4. **Create endpoint**

### 3. After Creation
- **Manage** > **Edit Endpoint** to adjust GPU, timeout, etc.
- No network volume needed — models download from HuggingFace

### 4. Test Locally
Create `.env` file:
```
RUNPOD_API_KEY=your_api_key_here
```

```bash
pip install requests python-dotenv
python test_endpoint.py
```

---

## Endpoint Settings Per Model

### LTX 2B (`2b-model` branch)
| Setting | Value |
|---|---|
| GPU | 24 GB PRO |
| Container disk | 20 GB |
| Max workers | 3 |
| Active workers | 0 |
| Idle timeout | 30-120 sec |
| Execution timeout | 900 sec |
| Cold start | ~3-4 min |

### LTX 13B (`main` branch)
| Setting | Value |
|---|---|
| GPU | 96 GB (1st), 80 GB PRO (2nd) |
| Container disk | 60 GB |
| Max workers | 3 |
| Active workers | 0 |
| Idle timeout | 30-120 sec |
| Execution timeout | 1800 sec |
| Cold start | ~10-15 min |

### Wan2.1 14B (`wan2.1` branch)
| Setting | Value |
|---|---|
| GPU | 96 GB (1st), 80 GB PRO (2nd) |
| Container disk | 100 GB |
| Max workers | 3 |
| Active workers | 0 |
| Idle timeout | 30-120 sec |
| Execution timeout | 1800 sec |
| Cold start | ~15-20 min |

### LTX-2.3 22B (`ltx-2.3` branch)
| Setting | Value |
|---|---|
| GPU | 96 GB (1st), 80 GB PRO (2nd) |
| Container disk | 100 GB |
| Max workers | 3 |
| Active workers | 0 |
| Idle timeout | 30-120 sec |
| Execution timeout | 1800 sec |
| Env variable | HF_TOKEN (required for Gemma text encoder) |
| Cold start | ~15-20 min |

---

## API Usage

### LTX 2B — Text-to-Video
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

### LTX 2B — Image-to-Video
```json
{
  "input": {
    "image": "<base64-or-url>",
    "prompt": "The scene comes to life",
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

### LTX 13B — Text-to-Video (with upscaling)
```json
{
  "input": {
    "prompt": "A woman smiling in golden sunset light, cinematic",
    "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
    "num_frames": 97,
    "fps": 24,
    "width": 832,
    "height": 480,
    "num_inference_steps": 50,
    "guidance_scale": 3.5,
    "upscale": true,
    "denoise_strength": 0.5
  }
}
```

### LTX 13B — Image-to-Video
```json
{
  "input": {
    "image": "<base64-or-url>",
    "prompt": "Smoke billows upward, flames crackling",
    "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
    "num_frames": 97,
    "fps": 24,
    "width": 832,
    "height": 480,
    "num_inference_steps": 50,
    "guidance_scale": 3.5,
    "upscale": true,
    "denoise_strength": 0.5
  }
}
```

### Wan2.1 14B — Text-to-Video
```json
{
  "input": {
    "prompt": "A woman smiling in golden sunset light, cinematic, realistic",
    "num_frames": 81,
    "fps": 16,
    "width": 832,
    "height": 480,
    "num_inference_steps": 30,
    "guidance_scale": 5.0
  }
}
```

### LTX-2.3 22B — Text-to-Video
```json
{
  "input": {
    "prompt": "A woman smiling in golden sunset light, cinematic, realistic, detailed skin texture",
    "num_frames": 121,
    "fps": 25,
    "width": 768,
    "height": 512,
    "num_inference_steps": 40,
    "cfg_scale": 3.0,
    "stg_scale": 1.0
  }
}
```

---

## Input Parameters

### LTX 2B
| Parameter | Default | Description |
|---|---|---|
| `prompt` | `""` | Text description |
| `image` | `null` | Image URL or base64 |
| `negative_prompt` | `"worst quality..."` | What to avoid |
| `num_frames` | `49` | Frames (~2 sec at 24fps) |
| `fps` | `24` | Frames per second |
| `width` | `768` | Width (divisible by 32) |
| `height` | `512` | Height (divisible by 32) |
| `num_inference_steps` | `50` | Denoising steps |
| `guidance_scale` | `3.0` | Prompt adherence |
| `decode_timestep` | `0.03` | VAE decode (critical) |
| `decode_noise_scale` | `0.025` | VAE noise (critical) |
| `seed` | `-1` | Random seed |

### LTX 13B
| Parameter | Default | Description |
|---|---|---|
| `prompt` | `""` | Text description |
| `image` | `null` | Image URL or base64 |
| `negative_prompt` | `"worst quality..."` | What to avoid |
| `num_frames` | `97` | Frames (~4 sec at 24fps) |
| `fps` | `24` | Frames per second |
| `width` | `832` | Output width |
| `height` | `480` | Output height |
| `num_inference_steps` | `30` | Denoising steps |
| `guidance_scale` | `5.0` | Prompt adherence |
| `upscale` | `true` | Enable 3-step upscale |
| `denoise_strength` | `0.4` | Upscale denoising |
| `seed` | `-1` | Random seed |

### Wan2.1 14B
| Parameter | Default | Description |
|---|---|---|
| `prompt` | `""` | Text description |
| `image` | `null` | Image URL or base64 |
| `num_frames` | `81` | Frames (~5 sec at 16fps) |
| `fps` | `16` | Frames per second |
| `width` | `832` | Width |
| `height` | `480` | Height |
| `num_inference_steps` | `30` | Denoising steps |
| `guidance_scale` | `5.0` | Prompt adherence |
| `seed` | `-1` | Random seed |

### LTX-2.3 22B
| Parameter | Default | Description |
|---|---|---|
| `prompt` | `""` | Text description |
| `num_frames` | `121` | Frames (~5 sec at 25fps) |
| `fps` | `25` | Frames per second |
| `width` | `768` | Width (divisible by 32) |
| `height` | `512` | Height (divisible by 32) |
| `num_inference_steps` | `40` | Denoising steps |
| `cfg_scale` | `3.0` | Guidance scale |
| `stg_scale` | `1.0` | Temporal guidance |
| `seed` | `-1` | Random seed |

---

## Cost Estimates (35 clips per batch)

| Model | Per clip (est.) | 35 clips | Cold start |
|---|---|---|---|
| LTX 2B (24GB) | ~$0.01 | ~$0.50 | ~$0.06 |
| LTX 13B (80GB) | ~$0.23 | ~$8.00 | ~$0.50 |
| Wan2.1 (80GB) | ~$0.30 | ~$10.50 | ~$0.50 |
| LTX-2.3 (96GB) | ~$0.25 | ~$8.75 | ~$0.70 |

### Billing Rules
- **Charged**: Worker running (cold start + execution + idle timeout)
- **NOT charged**: Build time, queue/delay time, no workers running
- Container disk is included in GPU rate (no extra charge)
- No storage costs (no network volume)

---

## Key Troubleshooting

### Quality is garbled/noisy (LTX 2B)
Include `decode_timestep=0.03` and `decode_noise_scale=0.025` in every request.

### CUDA out of memory (LTX 2B on 24GB)
Use `enable_sequential_cpu_offload()`. Max resolution: 768x512.

### Resolution must be divisible by 32
Both width and height must be divisible by 32.

### Network volume locks you to one datacenter
Don't use network volumes. Download from HuggingFace on cold start instead.

### Gated model error (LTX-2.3 / Gemma)
Accept the license at huggingface.co/google/gemma-3-4b-pt and add `HF_TOKEN` env variable to the endpoint.

### Low GPU availability
Select multiple GPU sizes as fallback options. Prioritize High Supply GPUs.

### RTX 5090 not compatible with PyTorch 2.4.0
Use base image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (PyTorch 2.8.0).

### Cold start too slow
Increase idle timeout to 120 sec when batching clips to keep worker warm between requests.

---

## Model Comparison

| Aspect | LTX 2B | LTX 13B | Wan2.1 | LTX-2.3 |
|---|---|---|---|---|
| Quality | Good | Better | Most realistic | Best (fast+quality) |
| Speed | Fastest | Fast | Slowest | Fast |
| Max res tested | 768x512 | 1280x736 | 832x480 | 768x512 |
| Text-to-video | Yes | Yes | Yes | Yes |
| Image-to-video | Yes | Yes | Yes | Coming soon |
| Audio generation | No | No | No | Yes |
| Upscaling | No | Built-in | No | Built-in |
| Framework | Diffusers | Diffusers | Diffusers | Custom (ltx-pipelines) |
