# Cấu hình chung – bám theo thiết lập trong bài báo HyDa-Gen (Nguyen et al., 2026)
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
BASE_DIR = DATA / "base"          # 20 ảnh gốc/lớp
TEST_DIR = DATA / "test"          # 100 ảnh test/lớp, tách biệt hoàn toàn
TDA_DIR = DATA / "tda"            # 4 bản tăng cường truyền thống/ảnh gốc
GDA_DIR = DATA / "gda"            # 4 bản Stable Diffusion/ảnh gốc
GALLERY_DIR = DATA / "gallery"    # ảnh so sánh Strength cho app/video
MODEL_DIR = ROOT / "models"

# Tên thư mục lớp trong PlantVillage -> tên hiển thị tiếng Việt
CLASSES = {
    "Tomato___Early_blight": "Bệnh đốm vòng (Early blight)",
    "Tomato___Late_blight": "Bệnh mốc sương (Late blight)",
    "Tomato___Leaf_Mold": "Bệnh nấm lá (Leaf mold)",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Virus xoăn vàng lá (YLCV)",
    "Tomato___healthy": "Lá khỏe mạnh (Healthy)",
}
CLASS_NAMES = list(CLASSES.keys())

# Mô tả triệu chứng (Mục 3.1 của bài báo) – hiển thị trong app
SYMPTOMS = {
    "Tomato___Early_blight": "Đốm nâu sẫm đến đen, thường tròn, có vòng đồng tâm, viền vàng.",
    "Tomato___Late_blight": "Vết bệnh lớn, không đều, màu nâu đen, úng nước; ẩm cao có lớp mốc trắng mặt dưới lá.",
    "Tomato___Leaf_Mold": "Đốm vàng mặt trên lá, mặt dưới có lớp mốc xanh ô liu hoặc xám, ranh giới không rõ.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Lá vàng rõ, xoăn cong lên, phiến lá nhỏ; không có đốm hoại tử.",
    "Tomato___healthy": "Lá xanh đều, không đốm, không biến dạng.",
}

N_TRAIN, N_TEST, N_AUG = 20, 100, 4
IMG_SIZE = 224
SEED = 42
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Stable Diffusion Img2Img: cấu hình tốt nhất trên PlantVillage (GDA_2)
SD_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"  # repo runwayml/... đã bị gỡ khỏi Hugging Face (2024)
SD_STRENGTH = 0.35
SD_GUIDANCE = 7.5
SD_STEPS = 50

# Prompt đại diện (Bảng 2 của bài báo, sinh bởi Gemini 2.5 Flash)
PROMPTS = {
    "Tomato___Early_blight": "Tomato Early blight leaf, dark brown to black angular spots, often with concentric rings resembling a target, surrounded by yellow halo, dry papery texture, sunken lesions, macro photography, 4k",
    "Tomato___Late_blight": "Tomato Late blight leaf, large irregular water-soaked dark brown to black lesions, often with a pale green halo, fuzzy white fungal growth on the underside, macro photography, 4k",
    "Tomato___Leaf_Mold": "Tomato Leaf Mold leaf, irregular bright yellow spots on upper surface, correlating dense olive-green to purplish-brown fuzzy mold growth on lower surface, lesions often angularly delimited by veins, macro photography, 4k",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato YLCV leaf, severe interveinal chlorosis, bright yellow margins, upward leaf curling, puckered and thickened texture, no distinct spots or lesions, macro photography, 4k",
    "Tomato___healthy": "Healthy fresh tomato leaf, vibrant uniform green color, smooth surface, clearly visible veins, no spots or discoloration, highly detailed, photorealistic, macro photography, 4k",
}
