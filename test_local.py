import os
import cv2
import numpy as np
from engine import DepthPortraitEngine

IMAGE_PATH = "test.jpg"  # Test etmek istediğin standart fotoğrafın yolu

def create_dummy_image_if_missing(path: str):
    """Eğer test.jpg yoksa test amaçlı sentetik bir görüntü oluşturur."""
    if not os.path.exists(path):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(img, (0, 0), (640, 480), (60, 40, 30), -1)
        cv2.circle(img, (320, 240), 110, (40, 180, 240), -1)
        cv2.putText(img, "ON PLAN", (245, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
        cv2.imwrite(path, img)
        print(f"[BİLGİ] '{path}' bulunamadı, örnek bir test görseli oluşturuldu.")

def main():
    create_dummy_image_if_missing(IMAGE_PATH)
    image_bgr = cv2.imread(IMAGE_PATH)

    # 1. Motoru başlat ve derinlik haritasını bir kez çıkar
    engine = DepthPortraitEngine()
    depth_norm = engine.predict_depth(image_bgr)

    # 2. Pencereyi önce oluştur
    win_name = "Portre Test Ekrani"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # 3. Callback fonksiyonunu trackbar'lar oluşturulduktan sonra güvenli çalışacak şekilde tanımla
    def on_change(_):
        # Pencere kapatıldıysa hata vermesini önle
        try:
            thresh_val = cv2.getTrackbarPos("Odak Esigi", win_name)
            blur_val = cv2.getTrackbarPos("Bulaniklik", win_name)
            trans_val = cv2.getTrackbarPos("Gecis Yumusakligi", win_name)
        except cv2.error:
            return

        thresh = max(1, thresh_val) / 100.0
        blur = max(3, blur_val)
        trans = max(1, trans_val) / 100.0

        portrait, depth_col, _ = engine.apply_portrait_bokeh(
            image_bgr, depth_norm, threshold=thresh, blur_strength=blur, transition=trans
        )
        combined = np.hstack((image_bgr, depth_col, portrait))
        cv2.imshow(win_name, combined)

    # 4. Trackbar'ları oluştur (İsimler on_change içindekilerle birebir aynı olmalı)
    cv2.createTrackbar("Odak Esigi", win_name, 45, 100, on_change)
    cv2.createTrackbar("Bulaniklik", win_name, 35, 99, on_change)
    cv2.createTrackbar("Gecis Yumusakligi", win_name, 12, 50, on_change)

    # İlk görselleştirmeyi tetikle
    on_change(0)

    print("\n[BİLGİ] Test penceresi açıldı!")
    print("-> Kaydırıcılarla oynayarak odak ve bulanıklığı ayarlayabilirsin.")
    print("-> Kaydetmek için klavyeden 'S' tuşuna bas.")
    print("-> Çıkmak için 'Q' tuşuna bas.")

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord("q") or key == 27:  # 'q' veya ESC tuşu
            break
        elif key == ord("s"):  # 's' tuşu ile kaydet
            thresh = cv2.getTrackbarPos("Odak Esigi", win_name) / 100.0
            blur = cv2.getTrackbarPos("Bulaniklik", win_name)
            trans = max(1, cv2.getTrackbarPos("Gecis Yumusakligi", win_name)) / 100.0
            
            portrait, depth_col, _ = engine.apply_portrait_bokeh(
                image_bgr, depth_norm, threshold=thresh, blur_strength=blur, transition=trans
            )
            cv2.imwrite("cikti_portre.jpg", portrait)
            cv2.imwrite("cikti_derinlik.jpg", depth_col)
            print("[BAŞARILI] 'cikti_portre.jpg' ve 'cikti_derinlik.jpg' başarıyla kaydedildi!")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()