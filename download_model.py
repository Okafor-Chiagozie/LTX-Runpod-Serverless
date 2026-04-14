"""
Run this on a temporary RunPod GPU pod with a network volume attached.
The model downloads fast on RunPod's network (~1-2 minutes).
"""
from huggingface_hub import snapshot_download

snapshot_download(
    "Lightricks/LTX-Video",
    local_dir="/runpod-volume/ltx-model",
    allow_patterns=[
        "model_index.json",
        "scheduler/*",
        "text_encoder/*",
        "tokenizer/*",
        "transformer/*",
        "vae/*",
    ],
)
print("Done! Model saved to /runpod-volume/ltx-model")
