import runpod
import torch
import numpy as np
import base64
import tempfile
import os
import requests
from io import BytesIO
from PIL import Image

print("Loading Wan2.1 14B pipeline...")

from diffusers import AutoencoderKLWan, WanPipeline, WanImageToVideoPipeline
from diffusers.utils import export_to_video
from transformers import CLIPVisionModel

T2V_MODEL = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
I2V_MODEL = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"

# Load text-to-video pipeline only at startup (lighter on VRAM)
print("Loading text-to-video pipeline...")
t2v_vae = AutoencoderKLWan.from_pretrained(T2V_MODEL, subfolder="vae", torch_dtype=torch.float32)
t2v_pipe = WanPipeline.from_pretrained(T2V_MODEL, vae=t2v_vae, torch_dtype=torch.bfloat16)
t2v_pipe.to("cuda")

# Image-to-video pipeline loaded on demand to save VRAM
i2v_pipe = None

def get_i2v_pipe():
    global i2v_pipe
    if i2v_pipe is None:
        print("Loading image-to-video pipeline on demand...")
        # Move t2v to CPU to free VRAM
        t2v_pipe.to("cpu")
        torch.cuda.empty_cache()
        i2v_image_encoder = CLIPVisionModel.from_pretrained(I2V_MODEL, subfolder="image_encoder", torch_dtype=torch.float32)
        i2v_vae = AutoencoderKLWan.from_pretrained(I2V_MODEL, subfolder="vae", torch_dtype=torch.float32)
        i2v_pipe = WanImageToVideoPipeline.from_pretrained(I2V_MODEL, vae=i2v_vae, image_encoder=i2v_image_encoder, torch_dtype=torch.bfloat16)
        i2v_pipe.to("cuda")
    return i2v_pipe

print("Wan2.1 text-to-video pipeline loaded and ready.")

DEFAULT_NEGATIVE = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"


def load_image_from_input(image_input):
    if image_input.startswith("http://") or image_input.startswith("https://"):
        response = requests.get(image_input, timeout=30)
        return Image.open(BytesIO(response.content)).convert("RGB")
    elif image_input.startswith("data:image"):
        header, data = image_input.split(",", 1)
        return Image.open(BytesIO(base64.b64decode(data))).convert("RGB")
    else:
        return Image.open(BytesIO(base64.b64decode(image_input))).convert("RGB")


def video_to_base64(video_path):
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def handler(job):
    try:
        inputs = job["input"]

        prompt          = inputs.get("prompt", "")
        negative_prompt = inputs.get("negative_prompt", DEFAULT_NEGATIVE)
        num_frames      = inputs.get("num_frames", 81)
        fps             = inputs.get("fps", 16)
        width           = inputs.get("width", 832)
        height          = inputs.get("height", 480)
        num_steps       = inputs.get("num_inference_steps", 30)
        guidance_scale  = inputs.get("guidance_scale", 5.0)
        seed            = inputs.get("seed", -1)

        generator = None if seed == -1 else torch.Generator("cuda").manual_seed(seed)

        image_input = inputs.get("image")

        if image_input:
            pipe = get_i2v_pipe()
            image = load_image_from_input(image_input)
            # Resize maintaining aspect ratio within max area
            max_area = height * width
            aspect_ratio = image.height / image.width
            mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
            calc_height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
            calc_width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
            image = image.resize((calc_width, calc_height))

            print(f"Image-to-video: {num_frames} frames @ {fps}fps, {calc_width}x{calc_height}")
            output = pipe(
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=calc_height,
                width=calc_width,
                num_frames=num_frames,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).frames[0]
        elif prompt:
            print(f"Text-to-video: {num_frames} frames @ {fps}fps, {width}x{height}")
            output = t2v_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).frames[0]
        else:
            return {"error": "Provide 'image', 'prompt', or both."}

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        export_to_video(output, tmp_path, fps=fps)
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
