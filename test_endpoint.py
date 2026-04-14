import requests
import base64
import time
import sys
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("RUNPOD_API_KEY")
ENDPOINT_ID = "70nfpq27c08tfl"

BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

payload = {
    "input": {
        "prompt": "A woman with long brown hair smiles warmly in golden sunset light, close-up portrait, cinematic, realistic",
        "num_frames": 81,
        "fps": 16,
        "width": 832,
        "height": 480,
        "num_inference_steps": 30,
        "guidance_scale": 5.0,
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
