#  Akıllı Portre Servisi (3D Depth Estimation & Bokeh)

Bu proje, standart 2 boyutlu fotoğraflardan yapay zeka destekli piksel bazlı derinlik haritası (depth map) çıkararak, kullanıcının belirlediği odak eşiğine göre fotoğraflara dinamik "Portre Modu" (Bokeh efekti) uygulayan uçtan uca bir web servisidir. Standart bir model çıkarım (inference) ödevinden farklı olarak, yapay zeka modeli bir iş mantığıyla (business logic) birleştirilerek tam teşekküllü bir API'a dönüştürülmüştür.

##  Neler Kullanıldı?

* **Yapay Zeka Modeli:** Hugging Face `Depth-Anything-V2-Base-hf`
* **Derin Öğrenme Çerçevesi:** PyTorch (RTX 3060 Tensor Çekirdekleri ile `float16` hassasiyetinde optimize edilmiştir)
* **Görüntü İşleme ve Matris Operasyonları:** OpenCV, NumPy, Pillow
* **Backend (Sunucu):** FastAPI, Uvicorn
* **Frontend (Arayüz):** HTML5, CSS3, Vanilla JavaScript
* **Ortam Yönetimi:** `uv` (Python Sanal Ortam Yöneticisi)

##  Proje Sonucu ve Temel Özellikler

Bu proje sonucunda, kullanıcının internet tarayıcısı üzerinden fotoğraf yükleyip anlık olarak arka plan bulanıklığını ayarlayabileceği tam çalışan bir sistem elde edilmiştir:

1. **Uçtan Uca Mimari:** Model sadece konsolda çalışan bir kod bloğu değil; Backend (FastAPI) ve Frontend (HTML/JS) ile entegre bir ürün haline getirilmiştir.
2. **GPU & Performans Optimizasyonu:** Kullanıcı bir fotoğraf yüklediğinde derinlik tahmini GPU'da (CUDA) sadece **1 kez** yapılır ve belleğe (RAM) alınır. Kullanıcı arayüzdeki "Odak Eşiği", "Bulanıklık Şiddeti" veya "Geçiş Yumuşaklığı" kaydırıcılarını (slider) değiştirdiğinde model tekrar çalışmaz; derinlik haritası üzerinden saniyenin onda biri hızında OpenCV alfa harmanlama (alpha blending) ile anlık sonuç üretilir.
3. **Dinamik Odak Maskeleme:** Klasik arka plan silici araçların aksine, 3 boyutlu bağıl derinlik ($D \in [0, 1]$) hesaplanarak ön plan ile arka plan arasında doğal bir Gauss geçişi sağlanır.
4. **Yerel Test Aracı:** Web sunucusuna ihtiyaç duymadan, doğrudan masaüstünde OpenCV pencereleriyle hızlı deneme yapmak için bağımsız bir `test_local.py` motoru içerir.

##  Nasıl Çalıştırılır?

**1. Gereksinimleri Yükleyin:**
```bash
pip install -r requirements.txt
