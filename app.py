"""App demo: Chẩn đoán bệnh lá cà chua với ít dữ liệu – minh họa nghiên cứu HyDa-Gen.
Chạy: python app.py  -> mở http://127.0.0.1:7860 (thêm share=True nếu cần mở trên điện thoại)
"""
import json
from pathlib import Path
import torch, torch.nn as nn
import gradio as gr
from PIL import Image
from torchvision import models, transforms as T
from config import (CLASSES, CLASS_NAMES, SYMPTOMS, MODEL_DIR, GALLERY_DIR, IMG_SIZE,
                    IMAGENET_MEAN, IMAGENET_STD)

dev = "cuda" if torch.cuda.is_available() else "cpu"
tf = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
tda = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.RandomHorizontalFlip(0.5), T.RandomRotation(20),
                 T.ColorJitter(0.2, 0.2, 0.2, 0.1)])


def load_model():
    # Ưu tiên mô hình cấu hình chính của bài báo: CDA + đóng băng một phần; nếu chưa có thì TDA, rồi Baseline.
    # Không bao giờ ưu tiên mô hình thí nghiệm "từ đầu" (cfg2) – chỉ dùng để minh họa.
    rank = {"cda_cfg1": 0, "tda_cfg1": 1, "base_cfg1": 2, "cda_cfg3": 3, "tda_cfg3": 4, "base_cfg3": 5}
    ckpts = sorted(MODEL_DIR.glob("*.pt"), key=lambda p: (rank.get(p.stem, 9), p.stem))
    if not ckpts:
        raise FileNotFoundError("Chưa có mô hình trong models/. Hãy chạy 04_huan_luyen.py trước.")
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(CLASS_NAMES))
    m.load_state_dict(torch.load(ckpts[0], map_location=dev))
    return m.to(dev).eval(), ckpts[0].stem


MODEL, MODEL_TAG = load_model()


@torch.no_grad()
def diagnose(img):
    if img is None:
        return {}, "Hãy tải lên hoặc chụp ảnh một lá cà chua."
    prob = torch.softmax(MODEL(tf(img.convert("RGB")).unsqueeze(0).to(dev)), 1)[0].cpu()
    scores = {CLASSES[c]: float(p) for c, p in zip(CLASS_NAMES, prob)}
    top = CLASS_NAMES[int(prob.argmax())]
    note = (f"### {CLASSES[top]} – độ tin cậy {float(prob.max()):.0%}\n"
            f"**Triệu chứng đặc trưng:** {SYMPTOMS[top]}\n\n")
    if float(prob.max()) < 0.6:
        note += "⚠️ Độ tin cậy thấp – nên chụp lại lá rõ nét, trên nền sáng, đủ ánh sáng.\n\n"
    note += ("_Đây là nguyên mẫu học tập (huấn luyện 20 ảnh/lớp, dữ liệu phòng lab PlantVillage). "
             "Kết quả chỉ mang tính tham khảo – hãy hỏi ý kiến cán bộ khuyến nông trước khi xử lý._")
    return scores, note


def tda_preview(img):
    return [tda(img.convert("RGB")) for _ in range(4)] if img is not None else []


def gallery(cls_label):
    cls = CLASS_NAMES[list(CLASSES.values()).index(cls_label)]
    d = GALLERY_DIR / cls
    caps = {"0_goc": "Ảnh gốc", "sd_strength_0.35": "Strength 0.35 – giữ triệu chứng",
            "sd_strength_0.50": "Strength 0.50", "sd_strength_0.65": "Strength 0.65 – biến dạng"}
    return [(str(p), caps.get(p.stem, p.stem)) for p in sorted(d.glob("*.jpg"))] if d.exists() else []


def results_table():
    rows = []
    for f in sorted(MODEL_DIR.glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        rows.append([r["data"].upper(), r["config"], r["n_train"], r["test_accuracy"], r["test_macro_f1"]])
    return rows


SCIENCE_MD = """
### Luận cứ khoa học
**Nguồn:** Nguyen T.T., Tran C.L.H., Pham N.H., Nguyen H.T. (2026). *Generative AI-driven hybrid data augmentation
for robust tomato leaf disease classification in low-data regimes.* Multimedia Tools and Applications 85:728 (Springer).
DOI: 10.1007/s11042-026-21895-2

| Phát hiện chính (PlantVillage, 20 ảnh/lớp, 15 fold) | Kết quả |
|---|---|
| Huấn luyện từ đầu, không tăng cường | 50,19% ± 19,06 (4/15 fold "sụp đổ") |
| Huấn luyện từ đầu + tăng cường lai CDA | **72,13% ± 3,00** (+21,95 điểm %, p_holm = 0,009) |
| Pretrained, đóng băng một phần: Baseline → CDA | 90,44% → 91,40% |
| Stable Diffusion Strength 0,35 | Giữ được hiệu năng ngang Baseline |
| Strength ≥ 0,50 | Giảm hiệu năng có ý nghĩa thống kê (ảnh bị "trôi ngữ nghĩa") |
| Dữ liệu ngoài đồng (PDR2018) | Tăng cường không giúp, có cấu hình còn làm giảm → cần dữ liệu thực địa |
"""

with gr.Blocks(title="Lá Cà Chua AI") as demo:
    gr.Markdown(f"# 🍅 Lá Cà Chua AI – chẩn đoán bệnh với chỉ 20 ảnh/lớp\n"
                f"Nhóm SV CNTT – FPT Polytechnic Cần Thơ · GVHD: Nguyễn Thế Trung · Mô hình: `{MODEL_TAG}` · Thiết bị: `{dev}`")
    with gr.Tab("1. Chẩn đoán"):
        with gr.Row():
            inp = gr.Image(type="pil", sources=["upload", "webcam"], label="Ảnh lá cà chua")
            with gr.Column():
                out_lbl = gr.Label(num_top_classes=3, label="Kết quả")
                out_md = gr.Markdown()
        gr.Button("Chẩn đoán", variant="primary").click(diagnose, inp, [out_lbl, out_md])
    with gr.Tab("2. AI tạo sinh tăng cường dữ liệu"):
        gr.Markdown("**Tăng cường truyền thống (TDA):** xoay, lật, đổi màu – không tạo thông tin mới.")
        inp2 = gr.Image(type="pil", label="Ảnh gốc")
        gal_tda = gr.Gallery(columns=4, height=220, label="4 bản TDA")
        gr.Button("Tạo 4 bản TDA").click(tda_preview, inp2, gal_tda)
        gr.Markdown("**Stable Diffusion Img2Img:** Strength càng cao, ảnh càng khác ảnh gốc. "
                    "Bài báo chỉ ra 0,35 là mức an toàn; 0,65 làm biến dạng triệu chứng bệnh.")
        sel = gr.Dropdown(list(CLASSES.values()), value=list(CLASSES.values())[0], label="Chọn lớp bệnh")
        gal_sd = gr.Gallery(columns=4, height=260, label="So sánh Strength")
        sel.change(gallery, sel, gal_sd)
        demo.load(gallery, sel, gal_sd)
    with gr.Tab("3. Luận cứ khoa học"):
        gr.Markdown(SCIENCE_MD)
        gr.Markdown("### Kết quả nhóm tự chạy lại (1 lần chạy, tập test 100 ảnh/lớp)")
        tbl = gr.Dataframe(headers=["Dữ liệu", "Cấu hình", "Số ảnh train", "Accuracy (%)", "Macro-F1 (%)"])
        demo.load(results_table, None, tbl)

if __name__ == "__main__":
    demo.launch()
