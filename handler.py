import runpod
import torch
import base64
import tempfile
import os
from huggingface_hub import hf_hub_download

print("Loading LTX-2.3 22B pipeline...")

from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_core.components.guiders import MultiModalGuiderParams

# Download model files
MODEL_DIR = "/app/models"
os.makedirs(MODEL_DIR, exist_ok=True)

REPO_ID = "Lightricks/LTX-2.3-fp8"

def download_if_needed(filename):
    local_path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(local_path):
        print(f"Downloading {filename}...")
        hf_hub_download(REPO_ID, filename=filename, local_dir=MODEL_DIR)
    return local_path

checkpoint_path = download_if_needed("ltx-2.3-22b-distilled-fp8.safetensors")

# Upsampler is in the main repo, not fp8
upsampler_local = os.path.join(MODEL_DIR, "ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
if not os.path.exists(upsampler_local):
    print("Downloading spatial upsampler...")
    hf_hub_download("Lightricks/LTX-2.3", filename="ltx-2.3-spatial-upscaler-x2-1.1.safetensors", local_dir=MODEL_DIR)
upsampler_path = upsampler_local

# Download Gemma text encoder
GEMMA_REPO = "Lightricks/gemma-3-12b-it-qat-q4_0-unquantized"
gemma_dir = os.path.join(MODEL_DIR, "gemma")
gemma_config = os.path.join(gemma_dir, "config.json")
if not os.path.exists(gemma_config):
    print("Downloading Gemma text encoder...")
    from huggingface_hub import snapshot_download
    snapshot_download(GEMMA_REPO, local_dir=gemma_dir, ignore_patterns=["*.gguf"])

print("Initializing pipeline...")
pipeline = TI2VidTwoStagesPipeline(
    checkpoint_path=checkpoint_path,
    distilled_lora=[],
    spatial_upsampler_path=upsampler_path,
    gemma_root=gemma_dir,
    loras=[],
)
print("LTX-2.3 pipeline loaded and ready.")


def video_to_base64(video_path):
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def handler(job):
    try:
        inputs = job["input"]

        prompt          = inputs.get("prompt", "")
        num_frames      = inputs.get("num_frames", 121)
        fps             = inputs.get("fps", 25)
        width           = inputs.get("width", 768)
        height          = inputs.get("height", 512)
        num_steps       = inputs.get("num_inference_steps", 40)
        cfg_scale       = inputs.get("cfg_scale", 3.0)
        stg_scale       = inputs.get("stg_scale", 1.0)
        seed            = inputs.get("seed", -1)

        if seed == -1:
            seed = torch.randint(0, 2**32, (1,)).item()

        video_guider_params = MultiModalGuiderParams(
            cfg_scale=cfg_scale,
            stg_scale=stg_scale,
            rescale_scale=0.7,
            modality_scale=3.0,
            skip_step=0,
            stg_blocks=[29],
        )

        audio_guider_params = MultiModalGuiderParams(
            cfg_scale=7.0,
            stg_scale=1.0,
            rescale_scale=0.7,
            modality_scale=3.0,
            skip_step=0,
            stg_blocks=[29],
        )

        if not prompt:
            return {"error": "Provide a 'prompt'."}

        print(f"Text-to-video: {num_frames} frames @ {fps}fps, {width}x{height}")

        frames_iter, audio = pipeline(
            prompt=prompt,
            negative_prompt="",
            images=[],
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=float(fps),
            num_inference_steps=num_steps,
            video_guider_params=video_guider_params,
            audio_guider_params=audio_guider_params,
        )

        # Encode video using LTX-2's native encoder
        from ltx_pipelines.utils.media_io import encode_video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        encode_video(
            video=frames_iter,
            fps=fps,
            audio=audio,
            output_path=tmp_path,
            video_chunks_number=1,
        )

        video_b64 = video_to_base64(tmp_path)
        os.unlink(tmp_path)

        return {
            "video_base64": video_b64,
            "format": "mp4",
            "frames": num_frames,
            "fps": fps,
            "width": width,
            "height": height,
        }

    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
