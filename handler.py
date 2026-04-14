import runpod
import torch
import base64
import tempfile
import os
import requests
from io import BytesIO
from PIL import Image

print("Loading LTX Video 13B 0.9.8 pipeline...")

from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.utils import export_to_video, load_video

MODEL_ID = "Lightricks/LTX-Video-0.9.8-dev"
UPSCALER_ID = "Lightricks/ltxv-spatial-upscaler-0.9.8"

pipe = LTXConditionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
pipe_upsample = LTXLatentUpsamplePipeline.from_pretrained(UPSCALER_ID, vae=pipe.vae, torch_dtype=torch.bfloat16)
pipe.to("cuda")
pipe_upsample.to("cuda")
pipe.vae.enable_tiling()
print("13B pipeline loaded and ready.")


def round_to_nearest_resolution(height, width):
    height = height - (height % pipe.vae_spatial_compression_ratio)
    width = width - (width % pipe.vae_spatial_compression_ratio)
    return height, width


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
        negative_prompt = inputs.get("negative_prompt", "worst quality, inconsistent motion, blurry, jittery, distorted")
        num_frames      = inputs.get("num_frames", 97)
        fps             = inputs.get("fps", 24)
        width           = inputs.get("width", 832)
        height          = inputs.get("height", 480)
        num_steps       = inputs.get("num_inference_steps", 30)
        guidance_scale  = inputs.get("guidance_scale", 5.0)
        seed            = inputs.get("seed", -1)
        upscale         = inputs.get("upscale", True)
        denoise_strength = inputs.get("denoise_strength", 0.4)

        generator = None if seed == -1 else torch.Generator("cuda").manual_seed(seed)

        image_input = inputs.get("image")
        conditions = []

        if image_input:
            image = load_image_from_input(image_input)
            video_cond = load_video(export_to_video([image]))
            conditions = [LTXVideoCondition(video=video_cond, frame_index=0)]
            print(f"Image-to-video: {num_frames} frames @ {fps}fps, {width}x{height}")
        elif prompt:
            print(f"Text-to-video: {num_frames} frames @ {fps}fps, {width}x{height}")
        else:
            return {"error": "Provide 'image', 'prompt', or both."}

        # Step 1: Generate at lower resolution
        downscale_factor = 2 / 3
        downscaled_height = int(height * downscale_factor)
        downscaled_width = int(width * downscale_factor)
        downscaled_height, downscaled_width = round_to_nearest_resolution(downscaled_height, downscaled_width)

        latents = pipe(
            conditions=conditions if conditions else None,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=downscaled_width,
            height=downscaled_height,
            num_frames=num_frames,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            decode_timestep=0.05,
            image_cond_noise_scale=0.025,
            generator=generator,
            output_type="latent",
        ).frames

        if upscale:
            # Step 2: Upscale latents 2x
            upscaled_latents = pipe_upsample(
                latents=latents,
                output_type="latent",
            ).frames

            # Step 3: Denoise upscaled video
            upscaled_height, upscaled_width = downscaled_height * 2, downscaled_width * 2
            video = pipe(
                conditions=conditions if conditions else None,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=upscaled_width,
                height=upscaled_height,
                num_frames=num_frames,
                denoise_strength=denoise_strength,
                num_inference_steps=10,
                latents=upscaled_latents,
                decode_timestep=0.05,
                image_cond_noise_scale=0.025,
                generator=generator,
                output_type="pil",
            ).frames[0]

            # Step 4: Resize to expected resolution
            video = [frame.resize((width, height)) for frame in video]
        else:
            video = pipe(
                conditions=conditions if conditions else None,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=downscaled_width,
                height=downscaled_height,
                num_frames=num_frames,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                decode_timestep=0.05,
                image_cond_noise_scale=0.025,
                generator=generator,
                output_type="pil",
            ).frames[0]

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        export_to_video(video, tmp_path, fps=fps)
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
