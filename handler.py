import runpod
import torch
import base64
import tempfile
import os
import io
from PIL import Image

print("Loading LTX-2 19B pipeline...")

from diffusers import FlowMatchEulerDiscreteScheduler, LTX2ImageToVideoPipeline
from diffusers.pipelines.ltx2 import LTX2Pipeline, LTX2LatentUpsamplePipeline
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES
from diffusers.pipelines.ltx2.export_utils import encode_video

MODEL_ID = "Lightricks/LTX-2"
device = "cuda:0"

# Load text-to-video pipeline with CPU offload
pipe = LTX2Pipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload(device=device)

# Load image-to-video pipeline sharing components
i2v_pipe = LTX2ImageToVideoPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
i2v_pipe.enable_sequential_cpu_offload(device=device)

# Load latent upsampler
latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
    MODEL_ID, subfolder="latent_upsampler", torch_dtype=torch.bfloat16
)
upsample_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=latent_upsampler)
upsample_pipe.enable_model_cpu_offload(device=device)

# Load distilled LoRA for stage 2
pipe.load_lora_weights(MODEL_ID, adapter_name="stage_2_distilled", weight_name="ltx-2-19b-distilled-lora-384.safetensors")
pipe.set_adapters("stage_2_distilled", 0.0)

print("LTX-2 pipeline loaded and ready.")


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
        width = inputs.get("width", 768)
        height = inputs.get("height", 512)
        num_steps = inputs.get("num_inference_steps", 40)
        guidance_scale = inputs.get("guidance_scale", 4.0)
        seed = inputs.get("seed", -1)
        upscale = inputs.get("upscale", True)
        image_b64 = inputs.get("image", None)

        if not prompt:
            return {"error": "Provide a 'prompt'."}

        if seed == -1:
            seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device="cpu").manual_seed(seed)

        # Determine which pipeline to use
        if image_b64:
            image_bytes = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image = image.resize((width, height))
            active_pipe = i2v_pipe
            extra_kwargs = {"image": image}
            print(f"Image-to-video: {num_frames} frames @ {fps}fps, {width}x{height}")
        else:
            active_pipe = pipe
            extra_kwargs = {}
            print(f"Text-to-video: {num_frames} frames @ {fps}fps, {width}x{height}")

        # Stage 1: Generate latents
        pipe.set_adapters("stage_2_distilled", 0.0)

        video_latent, audio_latent = active_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            frame_rate=float(fps),
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="latent",
            return_dict=False,
            **extra_kwargs,
        )

        if upscale:
            # Upscale latents 2x
            upscaled_video_latent = upsample_pipe(
                latents=video_latent,
                output_type="latent",
                return_dict=False,
            )[0]

            # Stage 2: Refine with distilled LoRA
            pipe.set_adapters("stage_2_distilled", 1.0)
            pipe.vae.enable_tiling()

            old_scheduler = pipe.scheduler
            new_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                pipe.scheduler.config, use_dynamic_shifting=False, shift_terminal=None
            )
            pipe.scheduler = new_scheduler

            video, audio = pipe(
                latents=upscaled_video_latent,
                audio_latents=audio_latent,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=3,
                noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
                sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
                guidance_scale=1.0,
                output_type="np",
                return_dict=False,
            )

            pipe.scheduler = old_scheduler
        else:
            # No upscale - generate directly
            pipe.vae.enable_tiling()
            video, audio = active_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                frame_rate=float(fps),
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type="np",
                return_dict=False,
                **extra_kwargs,
            )

        # Save video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        audio_data = None
        audio_sr = None
        if audio is not None:
            try:
                a = audio[0] if isinstance(audio, (list, tuple)) else audio
                audio_data = a.float().cpu() if hasattr(a, 'cpu') else None
                audio_sr = pipe.vocoder.config.output_sampling_rate
            except Exception:
                pass

        v = video[0] if isinstance(video, (list, tuple)) else video
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
