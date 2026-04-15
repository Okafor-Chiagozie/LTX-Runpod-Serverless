import requests
import base64
import time
import sys
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("RUNPOD_API_KEY")
ENDPOINT_ID = "01o3qnos7l4iqi"

BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# Load image as base64
import base64 as b64module
with open("scene_10.png", "rb") as img_file:
    image_b64 = b64module.b64encode(img_file.read()).decode("utf-8")

payload = {
    "input": {
        "image": image_b64,
        "prompt": "Inside a flooding ship corridor, knee-deep murky water rising rapidly, crew members in naval uniforms desperately pulling ropes and bracing against the surge, dim warm overhead lights reflecting off the water surface, metal walls with round portholes, cinematic camera slowly tracking forward through the corridor, dramatic tension, photorealistic, 4K film quality",
        "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted, static, low quality, deformed",
        "num_frames": 121,
        "fps": 24,
        "width": 1280,
        "height": 736,
        "num_inference_steps": 30,
        "guidance_scale": 4.0,
    }
}

print("Submitting job...")
resp = requests.post(f"{BASE_URL}/run", json=payload, headers=HEADERS)
job = resp.json()
print(f"Response: {job}")
job_id = job["id"]
print(f"Job ID: {job_id}")

print("Waiting for result...")
while True:
    status_resp = requests.get(f"{BASE_URL}/status/{job_id}", headers=HEADERS)
    result = status_resp.json()
    status = result["status"]
    print(f"  Status: {status}")

    if status == "COMPLETED":
        video_b64 = result["output"]["video_base64"]
        video_bytes = base64.b64decode(video_b64)
        filename = "output.mp4"
        with open(filename, "wb") as f:
            f.write(video_bytes)
        print(f"Video saved to {filename} ({len(video_bytes) / 1024 / 1024:.1f} MB)")
        break
    elif status == "FAILED":
        print(f"Job failed: {result}")
        break
    else:
        time.sleep(5)
