import base64
import os
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from engine import DepthPortraitEngine

app = FastAPI(title="3B Derinlik Tahmini ve Akıllı Portre Web Servisi")
engine = DepthPortraitEngine()

# Aktif oturumdaki görüntüyü ve derinlik matrisini önbellekte tutar (Anlık düzeltme için)
SESSION_CACHE = {
    "image_bgr": None,
    "depth_norm": None,
}


def encode_image_to_base64(img_bgr: np.ndarray) -> str:
    _, buffer = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Derinlik haritasını RTX 3060 üzerinde çıkar ve önbelleğe kaydet
    depth_norm = engine.predict_depth(image_bgr)
    SESSION_CACHE["image_bgr"] = image_bgr
    SESSION_CACHE["depth_norm"] = depth_norm

    portrait, depth_colored, mask_vis = engine.apply_portrait_bokeh(image_bgr, depth_norm)
    return JSONResponse({
        "original": encode_image_to_base64(image_bgr),
        "depth": encode_image_to_base64(depth_colored),
        "mask": encode_image_to_base64(cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)),
        "portrait": encode_image_to_base64(portrait),
    })


@app.post("/api/load-sample")
async def load_sample_image():
    """Kullanıcı fotoğraf yüklemeden tek tıkla standart test fotoğrafını yükler."""
    sample_path = "test.jpg"
    if os.path.exists(sample_path):
        image_bgr = cv2.imread(sample_path)
    else:
        # Klasörde test.jpg yoksa sentetik derinlik sahnesi üret
        image_bgr = np.zeros((500, 700, 3), dtype=np.uint8)
        cv2.rectangle(image_bgr, (0, 0), (700, 500), (55, 45, 35), -1)
        cv2.circle(image_bgr, (220, 260), 90, (90, 140, 230), -1)
        cv2.circle(image_bgr, (460, 240), 120, (50, 200, 120), -1)

    depth_norm = engine.predict_depth(image_bgr)
    SESSION_CACHE["image_bgr"] = image_bgr
    SESSION_CACHE["depth_norm"] = depth_norm

    portrait, depth_colored, mask_vis = engine.apply_portrait_bokeh(image_bgr, depth_norm)
    return JSONResponse({
        "original": encode_image_to_base64(image_bgr),
        "depth": encode_image_to_base64(depth_colored),
        "mask": encode_image_to_base64(cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)),
        "portrait": encode_image_to_base64(portrait),
    })


@app.post("/api/adjust")
async def adjust_parameters(
    threshold: float = Form(0.45),
    blur_strength: int = Form(35),
    transition: float = Form(0.12),
    invert_focus: bool = Form(False),
):
    """Modeli tekrar çalıştırmadan önbellekteki derinlik haritasıyla anlık düzeltme yapar."""
    if SESSION_CACHE["image_bgr"] is None or SESSION_CACHE["depth_norm"] is None:
        return JSONResponse({"error": "Önce bir fotoğraf yükleyin."}, status_code=400)

    portrait, _, mask_vis = engine.apply_portrait_bokeh(
        SESSION_CACHE["image_bgr"],
        SESSION_CACHE["depth_norm"],
        threshold=threshold,
        blur_strength=blur_strength,
        transition=transition,
        invert_focus=invert_focus,
    )
    return JSONResponse({
        "mask": encode_image_to_base64(cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)),
        "portrait": encode_image_to_base64(portrait),
    })