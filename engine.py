import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


class DepthPortraitEngine:

    def __init__(self, model_id: str = "depth-anything/Depth-Anything-V2-Base-hf"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        print(f"[BİLGİ] Model yükleniyor: {model_id} | Cihaz: {self.device.upper()} ({self.dtype})")
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(
            model_id, torch_dtype=self.dtype
        ).to(self.device)
        self.model.eval()

    def predict_depth(self, image_bgr: np.ndarray) -> np.ndarray:
        """BGR görüntüyü alır, [0, 1] aralığında normalize edilmiş derinlik matrisi döner."""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)

        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device, self.dtype)

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Orijinal görüntü boyutuna yeniden örnekleme (Bicubic Interpolation)
        h, w = image_bgr.shape[:2]
        prediction = F.interpolate(
            predicted_depth.unsqueeze(1).float(),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth = prediction.cpu().numpy()
        # [0, 1] aralığına Min-Max normalizasyonu (1 = En Yakın, 0 = En Uzak)
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth_norm

    @staticmethod
    def apply_portrait_bokeh(
        image_bgr: np.ndarray,
        depth_norm: np.ndarray,
        threshold: float = 0.45,
        blur_strength: int = 35,
        transition: float = 0.12,
        invert_focus: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Derinlik haritasını kullanarak arka planı (veya ön planı) bulanıklaştırır.
        Dönüş: (Portre_Görüntüsü, Renkli_Derinlik_Haritası, Alfa_Maskesi)
        """
        # Gauss çekirdek boyutu tek sayı (odd) olmalıdır
        ksize = int(blur_strength)
        if ksize % 2 == 0:
            ksize += 1
        ksize = max(3, ksize)

        # Yumuşak geçişli (feathered) odak maskesi hesaplama
        transition = max(0.01, float(transition))
        mask = np.clip((depth_norm - threshold) / transition + 0.5, 0.0, 1.0)

        if invert_focus:
            mask = 1.0 - mask

        # Kenar geçişlerini daha doğal yapmak için maskeye hafif yumuşatma
        mask_smooth = cv2.GaussianBlur(mask, (15, 15), 0)
        alpha = mask_smooth[:, :, np.newaxis]  # (H, W, 1)

        # Bulanık görüntüyü üret ve orijinalle harmanla
        blurred_img = cv2.GaussianBlur(image_bgr, (ksize, ksize), 0)
        portrait_img = (alpha * image_bgr + (1.0 - alpha) * blurred_img).astype(np.uint8)

        # Görselleştirme için renkli derinlik haritası (Inferno colormap)
        depth_vis = (depth_norm * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        mask_vis = (mask_smooth * 255).astype(np.uint8)

        return portrait_img, depth_colored, mask_vis