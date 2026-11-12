"""Bước 2: Tăng cường truyền thống (TDA) offline – 4 bản/ảnh gốc (Mục 3.2.1 bài báo).
Tên file giữ tiền tố ảnh gốc ("<stem>_tdaK") để loại trừ khi ảnh gốc nằm trong tập validation.
"""
from PIL import Image
from torchvision import transforms as T
import torch
from config import CLASS_NAMES, BASE_DIR, TDA_DIR, N_AUG, IMG_SIZE, SEED

torch.manual_seed(SEED)
tda = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),            # resize trước mọi phép ngẫu nhiên
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(20),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
])

for cls in CLASS_NAMES:
    out = TDA_DIR / cls
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted((BASE_DIR / cls).iterdir()):
        img = Image.open(f).convert("RGB")
        for k in range(N_AUG):
            tda(img).save(out / f"{f.stem}_tda{k}.jpg", quality=95)
    print(f"{cls}: xong TDA")
