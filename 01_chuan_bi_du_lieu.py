"""Bước 1: Lấy mẫu 20 ảnh train + 100 ảnh test mỗi lớp từ PlantVillage (không trùng lặp).
Chạy: python 01_chuan_bi_du_lieu.py --src <thư mục PlantVillage>
Tự nhận diện tên thư mục theo cả 2 kiểu đặt tên phổ biến (Tomato_Early_blight / Tomato___Early_blight).
"""
import argparse, random, shutil
from pathlib import Path
from config import CLASS_NAMES, BASE_DIR, TEST_DIR, N_TRAIN, N_TEST, SEED

KEYWORDS = {  # từ khóa nhận diện thư mục lớp (so khớp chữ thường, bỏ dấu "_")
    "Tomato___Early_blight": "earlyblight",
    "Tomato___Late_blight": "lateblight",
    "Tomato___Leaf_Mold": "leafmold",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "curlvirus",
    "Tomato___healthy": "healthy",
}

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="Thư mục gốc PlantVillage")
args = ap.parse_args()

dirs = [d for d in Path(args.src).rglob("*") if d.is_dir() and "tomato" in d.name.lower()]
random.seed(SEED)
for cls in CLASS_NAMES:
    match = [d for d in dirs if KEYWORDS[cls] in d.name.lower().replace("_", "")]
    assert match, f"Không tìm thấy thư mục cho lớp {cls}"
    files = sorted(p for p in match[0].iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    assert len(files) >= N_TRAIN + N_TEST, f"Lớp {cls} không đủ ảnh"
    random.shuffle(files)
    for subset, dst_root in ((files[:N_TRAIN], BASE_DIR), (files[N_TRAIN:N_TRAIN + N_TEST], TEST_DIR)):
        dst = dst_root / cls
        dst.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(subset):
            shutil.copy(f, dst / f"{cls[9:13].lower()}_{i:03d}{f.suffix.lower()}")
    print(f"{cls} <- {match[0].name}: {N_TRAIN} train / {N_TEST} test")
