import runpod
import torch
import base64
import tempfile
import os
import io
from PIL import Image

print("Loading LTX-2 19B pipeline...")

from diffusers import LTX2Pipeline, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.ltx2.export_utils import encode_video
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES

MODEL_ID = "Lightricks/LTX-2"
device = "cuda:0"

# Load text-to-video pipeline with CPU offload
pipe = LTX2Pipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload(device=device)
pipe.vae.enable_tiling()

# Load distilled LoRA for fast 8-step inference
pipe.load_lora_weights(MODEL_ID, adapter_name="distilled", weight_name="ltx-2-19b-distilled-lora-384.safetensors")
pipe.set_adapters("distilled", 1.0)

# Switch scheduler for distilled mode
pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
    pipe.scheduler.config, use_dynamic_shifting=False, shift_terminal=None
)

# I2V pipeline loaded on demand
i2v_pipe = None

print("LTX-2 pipeline loaded and ready.")


def get_i2v_pipe():
    global i2v_pipe
    if i2v_pipe is None:
        from diffusers import LTX2ImageToVideoPipeline
        print("Loading image-to-video pipeline...")
        i2v_pipe = LTX2ImageToVideoPipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16,
            text_encoder=pipe.text_encoder,
            tokenizer=pipe.tokenizer,
            transformer=pipe.transformer,
            vae=pipe.vae,
            scheduler=pipe.scheduler,
        )
        i2v_pipe.enable_model_cpu_offload(device=device)
        i2v_pipe.vae.enable_tiling()
    return i2v_pipe


def video_to_base64(video_path):
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def handler(job):
    try:
        inputs = job["input"]

        prompt = inputs.get("prompt", "")
        negative_prompt = inputs.get("negative_prompt", "shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly, transition, static.")
        num_frames = inputs.get("num_frames", 121)
        fps = inputs.get("fps", 24)
        width = inputs.get("width", 1280)
        height = inputs.get("height", 736)
        num_steps = inputs.get("num_inference_steps", 8)
        guidance_scale = inputs.get("guidance_scale", 1.0)
        seed = inputs.get("seed", -1)
        image_b64 = inputs.get("image", None)

        if not prompt:
            return {"error": "Provide a 'prompt'."}

        if seed == -1:
            seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device="cpu").manual_seed(seed)

        # Choose pipeline
        if image_b64:
            image_bytes = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image = image.resize((width, height))
            active_pipe = get_i2v_pipe()
            extra_kwargs = {"image": image}
            print(f"Image-to-video: {num_frames} frames @ {fps}fps, {width}x{height}")
        else:
            active_pipe = pipe
            extra_kwargs = {}
            print(f"Text-to-video: {num_frames} frames @ {fps}fps, {width}x{height}")

        # Generate video (distilled: 8 steps with predefined sigmas)
        result = active_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            frame_rate=float(fps),
            num_inference_steps=num_steps,
            sigmas=DISTILLED_SIGMA_VALUES,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="np",
            return_dict=False,
            **extra_kwargs,
        )

        video, audio = result[0], result[1] if len(result) > 1 else None

        # Save video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        audio_data = None
        audio_sr = None
        if audio is not None:
            try:
                a = audio[0] if isinstance(audio, (list, tuple)) else audio
                if hasattr(a, 'cpu'):
                    audio_data = a.float().cpu()
                    audio_sr = pipe.vocoder.config.output_sampling_rate
            except Exception:
                pass

        v = video[0] if isinstance(video, (list, tuple)) else video
        # Remove batch dimension if present (5D -> 4D)
        if hasattr(v, 'ndim') and v.ndim == 5:
            v = v[0]
        encode_video(
            v,
            fps=fps,
            audio=audio_data,
            audio_sample_rate=audio_sr,
            output_path=tmp_path,
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
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


runpod.serverless.start({"handler": handler})
