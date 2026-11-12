"""Bước 3: Tăng cường bằng AI tạo sinh (GDA) – Stable Diffusion v1.5 Img2Img (Mục 3.2.2 bài báo).
  python 03_tang_cuong_sd.py                 # tạo 4 ảnh/ảnh gốc, Strength 0.35 -> data/gda
  python 03_tang_cuong_sd.py --gallery       # tạo bộ ảnh so sánh Strength 0.35 / 0.50 / 0.65 cho app & video
Chạy được trên RTX 3050 (4-6 GB VRAM) nhờ fp16 + attention slicing + CPU offload; ~10-30 giây/ảnh.
"""
import argparse
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, PNDMScheduler
from config import (CLASS_NAMES, BASE_DIR, GDA_DIR, GALLERY_DIR, PROMPTS, N_AUG, SEED,
                    SD_MODEL, SD_STRENGTH, SD_GUIDANCE, SD_STEPS)

ap = argparse.ArgumentParser()
ap.add_argument("--gallery", action="store_true", help="Chỉ tạo ảnh so sánh Strength")
args = ap.parse_args()

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(SD_MODEL, torch_dtype=torch.float16, safety_checker=None)
pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()
pipe.enable_model_cpu_offload()          # tiết kiệm VRAM cho GPU 4-6 GB
gen = torch.Generator("cpu").manual_seed(SEED)


def make(img, cls, strength, n):
    return pipe(prompt=[PROMPTS[cls]] * n, image=img, strength=strength, guidance_scale=SD_GUIDANCE,
                num_inference_steps=SD_STEPS, generator=gen).images


for cls in CLASS_NAMES:
    files = sorted((BASE_DIR / cls).iterdir())
    if args.gallery:
        # 1 ảnh gốc/lớp, so sánh 3 mức Strength: 0.35 giữ triệu chứng, 0.65 biến dạng (semantic drift)
        src = Image.open(files[0]).convert("RGB").resize((512, 512))
        out = GALLERY_DIR / cls
        out.mkdir(parents=True, exist_ok=True)
        src.save(out / "0_goc.jpg")
        for s in (0.35, 0.50, 0.65):
            make(src, cls, s, 1)[0].save(out / f"sd_strength_{s:.2f}.jpg")
    else:
        out = GDA_DIR / cls
        out.mkdir(parents=True, exist_ok=True)
        for f in files:
            src = Image.open(f).convert("RGB").resize((512, 512))
            for k, im in enumerate(make(src, cls, SD_STRENGTH, N_AUG)):
                im.save(out / f"{f.stem}_sd{k}.jpg", quality=95)
    print(f"{cls}: xong")
