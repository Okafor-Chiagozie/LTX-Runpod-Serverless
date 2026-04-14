import requests
import base64
import time
import sys

API_KEY = "rpa_MI796DRR6PYBNINKYMJUFDMTG7LPR6BV5EN6RRLYyo6ah9"
ENDPOINT_ID = "swrgif95vdcviz"

BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

payload = {
    "input": {
        "prompt": "A cat walking through a sunny garden with flowers, cinematic lighting, high quality",
        "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
        "num_frames": 161,
        "fps": 24,
        "width": 1024,
        "height": 576,
        "num_inference_steps": 50,
        "decode_timestep": 0.03,
        "decode_noise_scale": 0.025,
    }
}

print("Submitting job...")
resp = requests.post(f"{BASE_URL}/run", json=payload, headers=HEADERS)
job = resp.json()
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
