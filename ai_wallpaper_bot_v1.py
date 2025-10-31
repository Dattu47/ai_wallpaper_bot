"""
AI Wallpaper Bot — v1 (Free, local)

What this script does:
- Generates a wallpaper image using Stable Diffusion (diffusers)
- Optionally upscales it with Real-ESRGAN if available
- Generates a short caption + hashtags with a local text model (distilgpt2)
- Uploads the image to Instagram using instagrapi

Notes (read before running):
- This is intended to be fully free / local. You must download the Stable Diffusion model weights
  the first time (Hugging Face accounts may be required for some models).
- A decent GPU (NVIDIA CUDA) is strongly recommended for reasonable inference times.
- Use a separate Instagram account for automated posting to reduce risk of being flagged.
- This script does a single generate-and-post run. Use cron / schedule to automate.

Files you should create alongside this script:
- config.json  (example below)
- requirements.txt (see bottom of this file)

----- Example config.json -----
{
  "instagram_username": "your_bot_username",
  "instagram_password": "your_password",
  "model_id": "runwayml/stable-diffusion-v1-5",
  "use_cuda": true,
  "prompt_templates": [
    "a dreamy night sky over snow-capped mountains, 4k wallpaper, ultra-detailed, cinematic",
    "golden hour ocean view with soft clouds, 4k wallpaper, high detail, minimal composition",
    "dense forest with light rays at sunrise, 4k wallpaper, ultra detailed, photorealistic"
  ],
  "hashtags": ["#wallpaper","#4k","#aesthetic","#nature","#mobilewallpaper"],
  "out_folder": "assets",
  "upscale": true
}
----- End config.json -----

Run:
1) Create & activate a virtualenv
2) pip install -r requirements.txt
3) python ai_wallpaper_bot_v1.py

"""

import os
import sys
import json
import random
import time
from datetime import datetime
from pathlib import Path
import schedule

# IMAGE AI imports
try:
    import torch
    from diffusers import StableDiffusionPipeline
except Exception as e:
    print("Missing diffusers/torch. Install requirements. Exiting.")
    raise

# UPSCALER (optional)
try:
    from realesrgan import RealESRGAN
    REAL_ESRGAN_AVAILABLE = True
except Exception:
    REAL_ESRGAN_AVAILABLE = False

# CAPTION LLM
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# INSTAGRAM
try:
    from instagrapi import Client
    INSTAGRAPI_AVAILABLE = True
except Exception:
    INSTAGRAPI_AVAILABLE = False

from PIL import Image

# ---------------------------
# Utility / helper functions
# ---------------------------

def load_config(path="config.json"):
    if not os.path.exists(path):
        print(f"Config file '{path}' not found. Create it using the example in the script header.")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_out_folder(folder):
    Path(folder).mkdir(parents=True, exist_ok=True)


# ---------------------------
# Image generation (Stable Diffusion)
# ---------------------------

def generate_image(prompt, model_id, use_cuda=True, height=1920, width=1080, guidance_scale=7.5, num_inference_steps=30):
    device = "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"
    print(f"Using device: {device}")

    # load pipeline (weights will be downloaded first time)
    # NOTE: you may need to set your HF token in environment or use a model that doesn't require authentication
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device=="cuda" else torch.float32)
    pipe = pipe.to(device)

    # simple generation; adjust arguments as needed
    print(f"Generating image for prompt: {prompt}")
    generator = None
    if device == "cuda":
        generator = torch.manual_seed(random.randint(1, 2**31 - 1))

    images = pipe(prompt, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images
    if not images:
        raise RuntimeError("Pipeline returned no images")
    img = images[0]

    # ensure aspect correct for phone wallpaper (portrait) by resizing/cropping
    img = img.convert("RGB")
    img = img.resize((width, height), resample=Image.LANCZOS)
    return img


# ---------------------------
# Upscale (Real-ESRGAN) — optional
# ---------------------------

def upscale_image_if_available(img, device="cpu"):
    if not REAL_ESRGAN_AVAILABLE:
        print("Real-ESRGAN not available, skipping upscaling.")
        return img
    try:
        model = RealESRGAN(device=device)
        # default weights name; you must download weights and place path or modify load_weights call
        # model.load_weights("weights/RealESRGAN_x4plus.pth")
        model.load_weights("RealESRGAN_x4plus.pth")
        sr = model.predict(img)
        return sr
    except Exception as e:
        print("Upscaling failed:", e)
        return img


# ---------------------------
# Caption + hashtag generation (local transformer)
# ---------------------------

def generate_caption(prompt_templates, hashtags, max_length=30):
    prompt = random.choice(prompt_templates)
    short_prompt = f"Write a short Instagram caption (1-2 lines) for: {prompt}"

    if TRANSFORMERS_AVAILABLE:
        try:
            gen = pipeline("text-generation", model="distilgpt2")
            out = gen(short_prompt, max_length=max_length, num_return_sequences=1)[0]["generated_text"]
            # heuristics: take the line after the query or the last sentence
            caption = out.strip()
            # ensure we don't repeat the query text
            if caption.lower().startswith(short_prompt.lower()):
                caption = caption[len(short_prompt):].strip()
            if not caption:
                caption = prompt
        except Exception as e:
            print("Local caption model failed, falling back to template.", e)
            caption = prompt
    else:
        print("transformers not available, using prompt as caption.")
        caption = prompt

    # append some hashtags (random subset)
    selected_hashtags = " ".join(random.sample(hashtags, min(4, len(hashtags))))
    return f"{caption}\n\n{selected_hashtags}"


# ---------------------------
# Instagram upload (instagrapi)
# ---------------------------

def upload_to_instagram(image_path, caption, username, password):
    if not INSTAGRAPI_AVAILABLE:
        print("instagrapi not installed. Cannot upload.")
        return False
    cl = Client()
    try:
        print("Logging in to Instagram...")
        cl.login(username, password)
        print("Uploading photo...")
        cl.photo_upload(image_path, caption)
        print("Upload finished.")
        return True
    except Exception as e:
        print("Instagram upload failed:", e)
        return False


# ---------------------------
# MAIN flow
# ---------------------------

def main():
    cfg = load_config()
    ensure_out_folder(cfg.get("out_folder", "assets"))

    prompt = random.choice(cfg.get("prompt_templates", ["ultra-detailed cinematic wallpaper, a stunning fusion of nature, futuristic technology,"
    " and abstract art — combining elements of mountains, galaxies, glowing neon lights, soft gradients, and surreal fantasy landscapes;"
    " balanced composition, dreamy lighting, elegant minimalism blended with high-concept sci-fi atmosphere, rendered in 8K ultra HD, "
    "realistic textures, vibrant yet harmonious color palette, deep shadows, and glowing highlights, perfect for wallpaper or Instagram aesthetic background"]))
    model_id = cfg.get("model_id", "runwayml/stable-diffusion-v1-5")
    use_cuda = cfg.get("use_cuda", True)

    # generate
    img = generate_image(prompt, model_id=model_id, use_cuda=use_cuda)

    # optional upscale
    if cfg.get("upscale", False):
        device = "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"
        img = upscale_image_if_available(img, device=device)

    # save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"wallpaper_{timestamp}.jpg"
    out_path = os.path.join(cfg.get("out_folder", "assets"), out_name)
    img.save(out_path, quality=95)
    print(f"Saved wallpaper to {out_path}")

    # caption
    caption = generate_caption(cfg.get("prompt_templates", []), cfg.get("hashtags", []))
    print("Caption prepared:\n", caption)

    # upload
    success = upload_to_instagram(out_path, caption, cfg.get("instagram_username"), cfg.get("instagram_password"))
    if not success:
        print("Upload failed or skipped. You can manually post the image located at:", out_path)




def run_scheduler():
    # Run every 6 hours (change as you like)
    schedule.every(6).hours.do(main)

    print("AI Wallpaper Bot scheduler started. Running every 6 hours...")

    while True:
        schedule.run_pending()
        time.sleep(60)  # check every minute


if __name__ == "__main__":
    # Uncomment one of the two options below:
    
    # 1️⃣ Run once and exit
     main()

    # 2️⃣ Run continuously on schedule
    #run_scheduler()



# ---------------------------
# requirements.txt content (for your convenience — put in a file named requirements.txt):
# ---------------------------
# torch
# diffusers
# transformers
# safetensors
# pillow
# instagrapi
# realesrgan  # optional
# schedule     # optional for scheduling


# Final notes:
# - Model downloads (diffusers) will happen the first time you run the pipeline; expect large downloads.
# - If you don't have GPU, set "use_cuda": false in config.json and be prepared for slow CPU generation.
# - Real-ESRGAN weights should be downloaded separately and the path adjusted in the code.
# - For reliability: create a dedicated Instagram account and don't post too frequently while testing.
