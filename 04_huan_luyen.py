"""Bước 4: Huấn luyện EfficientNet-B0 với 20 ảnh/lớp (Mục 3.3 & 4.2 bài báo).
  python 04_huan_luyen.py --data cda --config 1     # cấu hình chính: pretrained + đóng băng một phần
  python 04_huan_luyen.py --data base --config 2    # từ đầu, không tăng cường (dễ "sụp đổ")
  python 04_huan_luyen.py --data cda  --config 2    # từ đầu + CDA (minh họa hiệu ứng "giải cứu")
--data:   base | tda | cda (= gốc + TDA + Stable Diffusion)
--config: 1 = đóng băng features[0..5]; 2 = từ đầu (không pretrained); 3 = fine-tune toàn bộ
Lưu ý: bài báo dùng 15 fold + kiểm định Wilcoxon; script này chỉ chạy 1 lần để làm demo.
"""
import argparse, json, random
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms as T
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from config import (CLASS_NAMES, BASE_DIR, TEST_DIR, TDA_DIR, GDA_DIR, MODEL_DIR, IMG_SIZE, SEED,
                    IMAGENET_MEAN, IMAGENET_STD)

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="cda", choices=["base", "tda", "cda"])
ap.add_argument("--config", type=int, default=1, choices=[1, 2, 3])
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--val_ratio", type=float, default=0.2)
args = ap.parse_args()

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"
IMG_EXT = {".jpg", ".jpeg", ".png"}


def list_imgs(d):
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT) if d.exists() else []


# --- Chia train/val trên ẢNH GỐC; ảnh tăng cường sinh từ ảnh val bị loại để tránh rò rỉ dữ liệu ---
train, val = [], []
for y, cls in enumerate(CLASS_NAMES):
    orig = list_imgs(BASE_DIR / cls)
    random.shuffle(orig)
    n_val = max(1, int(len(orig) * args.val_ratio))
    val_stems = {p.stem for p in orig[:n_val]}
    val += [(p, y) for p in orig[:n_val]]
    train += [(p, y) for p in orig[n_val:]]
    aug_dirs = {"base": [], "tda": [TDA_DIR], "cda": [TDA_DIR, GDA_DIR]}[args.data]
    for d in aug_dirs:
        train += [(p, y) for p in list_imgs(d / cls) if p.stem.rsplit("_", 1)[0] not in val_stems]
test = [(p, y) for y, cls in enumerate(CLASS_NAMES) for p in list_imgs(TEST_DIR / cls)]
print(f"train={len(train)}  val={len(val)}  test={len(test)}  device={dev}")

norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
tf_train = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.RandomHorizontalFlip(0.5), T.RandomRotation(15),
                      T.ColorJitter(0.2, 0.2, 0.2), T.ToTensor(), norm])   # online aug nhẹ hơn offline
tf_eval = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), norm])


class ImgDS(Dataset):
    def __init__(self, items, tf): self.items, self.tf = items, tf
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), y


dl_tr = DataLoader(ImgDS(train, tf_train), batch_size=args.batch, shuffle=True, num_workers=0)  # Windows: giữ 0
dl_va = DataLoader(ImgDS(val, tf_eval), batch_size=32)
dl_te = DataLoader(ImgDS(test, tf_eval), batch_size=32)


def build_model(cfg):
    w = None if cfg == 2 else models.EfficientNet_B0_Weights.IMAGENET1K_V1
    m = models.efficientnet_b0(weights=w)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(CLASS_NAMES))
    if cfg == 1:  # đóng băng stem + 5 khối MBConv đầu (~0,85 triệu tham số)
        for blk in m.features[:6]:
            for p in blk.parameters():
                p.requires_grad = False
    return m


model = build_model(args.config).to(dev)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10)
crit = nn.CrossEntropyLoss()


@torch.no_grad()
def evaluate(dl):
    model.eval(); loss, ys, ps = 0.0, [], []
    for x, y in dl:
        out = model(x.to(dev)); loss += crit(out, y.to(dev)).item() * len(y)
        ys += y.tolist(); ps += out.argmax(1).cpu().tolist()
    return loss / len(dl.dataset), ys, ps


tag = f"{args.data}_cfg{args.config}"
MODEL_DIR.mkdir(exist_ok=True)
best, wait, patience = float("inf"), 0, 10
for ep in range(1, args.epochs + 1):
    model.train()
    for x, y in dl_tr:
        opt.zero_grad()
        crit(model(x.to(dev)), y.to(dev)).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    sched.step()
    vl, vy, vp = evaluate(dl_va)
    print(f"epoch {ep:02d}  val_loss={vl:.4f}  val_acc={accuracy_score(vy, vp):.3f}")
    if vl < best:                         # early stopping theo val loss
        best, wait = vl, 0
        torch.save(model.state_dict(), MODEL_DIR / f"{tag}.pt")
    else:
        wait += 1
        if wait >= patience:
            break

model.load_state_dict(torch.load(MODEL_DIR / f"{tag}.pt", map_location=dev))
_, ty, tp = evaluate(dl_te)
res = {"data": args.data, "config": args.config, "n_train": len(train),
       "test_accuracy": round(100 * accuracy_score(ty, tp), 2),
       "test_macro_f1": round(100 * f1_score(ty, tp, average="macro"), 2),
       "confusion_matrix": confusion_matrix(ty, tp).tolist()}
(MODEL_DIR / f"{tag}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(res, indent=2, ensure_ascii=False))
