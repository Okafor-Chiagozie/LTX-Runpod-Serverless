import runpod
import torch
import base64
import tempfile
import os
import requests
from io import BytesIO
from PIL import Image

print("Loading LTX Video 0.9.8 distilled pipeline...")

from diffusers import LTXPipeline, LTXImageToVideoPipeline
from diffusers.utils import export_to_video

MODEL_PATH = "/runpod-volume/ltx-model"

txt2vid_pipe = LTXPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
)
txt2vid_pipe.enable_sequential_cpu_offload()

# Share all components to avoid doubling memory usage
img2vid_pipe = LTXImageToVideoPipeline(**txt2vid_pipe.components)
print("Both pipelines loaded and ready.")


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
        num_frames      = inputs.get("num_frames", 161)
        fps             = inputs.get("fps", 24)
        width           = inputs.get("width", 1024)
        height          = inputs.get("height", 576)
        num_steps       = inputs.get("num_inference_steps", 50)
        guidance_scale  = inputs.get("guidance_scale", 3.0)
        decode_timestep = inputs.get("decode_timestep", 0.03)
        decode_noise_scale = inputs.get("decode_noise_scale", 0.025)
        seed            = inputs.get("seed", -1)

        generator = None if seed == -1 else torch.Generator("cuda").manual_seed(seed)

        image_input = inputs.get("image")

        if image_input:
            image = load_image_from_input(image_input)
            image = image.resize((width, height))
            print(f"Image-to-video: {num_frames} frames @ {fps}fps, {width}x{height}")
            output = img2vid_pipe(
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                decode_timestep=decode_timestep,
                decode_noise_scale=decode_noise_scale,
                generator=generator,
            )
        elif prompt:
            print(f"Text-to-video: {num_frames} frames @ {fps}fps, {width}x{height}")
            output = txt2vid_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                decode_timestep=decode_timestep,
                decode_noise_scale=decode_noise_scale,
                generator=generator,
            )
        else:
            return {"error": "Provide 'image', 'prompt', or both."}

        frames = output.frames[0]

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        export_to_video(frames, tmp_path, fps=fps)
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