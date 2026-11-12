# Lá Cà Chua AI – App demo cho cuộc thi “Chọn nghề, chọn tương lai” 2026

Nguyên mẫu chẩn đoán 5 lớp bệnh lá cà chua chỉ với **20 ảnh/lớp**, dựa trên bài báo:

> Nguyen T.T., Tran C.L.H., Pham N.H., Nguyen H.T. (2026). *Generative AI-driven hybrid data augmentation for robust
> tomato leaf disease classification in low-data regimes.* Multimedia Tools and Applications 85:728.
> https://doi.org/10.1007/s11042-026-21895-2

## Cài đặt
```bash
python -m venv .venv && .venv\Scripts\activate        # Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```
Tải PlantVillage (Kaggle: `emmarex/plantdisease`), giải nén, lấy thư mục chứa các folder `Tomato___*`.

## Chạy theo thứ tự
```bash
python 01_chuan_bi_du_lieu.py --src D:\PlantVillage   # 20 train + 100 test/lớp
python 02_tang_cuong_tda.py                           # 4 bản TDA/ảnh
python 03_tang_cuong_sd.py                            # 4 bản Stable Diffusion/ảnh (Strength 0.35)
python 03_tang_cuong_sd.py --gallery                  # ảnh so sánh Strength 0.35/0.50/0.65
python 04_huan_luyen.py --data cda  --config 1        # mô hình dùng trong app
python 04_huan_luyen.py --data base --config 2        # thí nghiệm minh họa: từ đầu, không tăng cường
python 04_huan_luyen.py --data cda  --config 2        # thí nghiệm minh họa: từ đầu + CDA
python app.py                                         # mở http://127.0.0.1:7860
```

## Lưu ý trung thực
- Kết quả trong app là **1 lần chạy** của nhóm; số liệu khoa học chính thức lấy từ bài báo (15 fold, kiểm định Wilcoxon + Holm).
- Mô hình học trên ảnh phòng lab → chụp lá trên **nền trơn, đủ sáng** khi demo. Với ảnh ngoài đồng, bài báo chỉ ra tăng cường dữ liệu chưa giúp được (domain gap).
- Ảnh do Stable Diffusion tạo ra phải được ghi chú “ảnh tạo bởi AI” khi đưa vào video và khai báo khi nộp bài.
