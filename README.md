# Smart Disaster Hub — ML Service

FastAPI tabanlı metin analizi servisi. Sosyal medya gönderilerinde **afet ilgisi (T1)**, **yardım talebi** ve **insani kategori (T2)** tespiti yapar.

**Smart Disaster Hub** projesinin Spring Boot backend'i bu servisi `http://localhost:8000` üzerinden çağırır.

---

## Özellikler

- **T1 — Disaster Relevance:** Metnin afet ile ilgili olup olmadığını sınıflandırır
- **T2 — Humanitarian:** Yardım talebi ve kategori (`urgent_needs`, `infrastructure_damage`, vb.)
- **Çok dilli destek:** FastText dil algılama + MarianMT çeviri (Türkçe → İngilizce)
- **İki model modu:** Logistic Regression (varsayılan, hızlı) ve RoBERTa (opsiyonel, daha yüksek doğruluk)

---

## Hızlı başlangıç

### Gereksinimler

- Python 3.10+
- ~2 GB disk (PyTorch + ilk çeviri modeli indirmesi)
- İnternet (ilk kurulumda model indirmesi için)

### 1. Repoyu klonla

```bash
git clone https://github.com/muratkeskin0/ml-service.git
cd ml-service
```

### 2. Sanal ortam ve bağımlılıklar

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Model dosyalarını indir

Bu repoda **kaynak kod** vardır; eğitilmiş model ağırlıkları **Hugging Face Hub**'da barındırılır (ücretsiz, public).

```bash
python scripts/download_models.py
```

İndirilenler:

| Dosya | Hub'da | Zorunlu (LR modu) |
|-------|--------|-------------------|
| `model.pkl`, `vectorizer.pkl` | Hayır | Evet |
| `char_vectorizer.pkl`, `feature_extractor.pkl`, `model_metadata.json` | Evet | Hayır |
| `roberta/` (ağırlıklar dahil) | Evet | Hayır (`use_roberta=true`) |

\* FastText yoksa basit heuristik dil algılama kullanılır.

Opsiyonel RoBERTa:

```bash
python scripts/download_models.py --roberta
```

### 4. Servisi başlat

```bash
cd inference
python app.py
```

Alternatif:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --app-dir inference
```

Tarayıcıda: [http://localhost:8000/health](http://localhost:8000/health)

---

## API uç noktaları

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| `GET` | `/health` | Servis durumu |
| `GET` | `/` | Servis bilgisi ve endpoint listesi |
| `POST` | `/t1/analyze` | Sadece T1 (afet ilgisi) |
| `POST` | `/t2/analyze` | Sadece T2 (yardım talebi + kategori) |
| `POST` | `/analyze` | T1 + T2 birlikte (önerilen) |

### Örnek istek

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"Deprem sonrası su ve barınma yardımına ihtiyacımız var.\", \"include_t2\": true}"
```

### Örnek yanıt

```json
{
  "is_disaster_related": true,
  "relevance_score": 0.91,
  "message": "Post afet ile ilgili (güven: 91.00%)",
  "model_used": "logistic_regression",
  "t2": {
    "is_help_request": true,
    "help_request_probability": 0.85,
    "humanitarian_labels": ["urgent_needs"],
    "category_probabilities": {
      "urgent_needs": 0.72,
      "infrastructure_damage": 0.08,
      "donations_volunteering": 0.05,
      "other": 0.15
    }
  }
}
```

RoBERTa kullanmak için isteğe `"use_roberta": true` ekleyin (model Hub'da mevcutsa).

---

## Model dosyaları nerede?

```
GitHub (bu repo)          → Python kodu, API, scriptler
Hugging Face Hub (ücretsiz) → model.pkl, vectorizer.pkl, RoBERTa ağırlıkları
Meta CDN (ücretsiz)       → FastText dil modeli
Hugging Face (runtime)    → MarianMT çeviri modeli (ilk çeviri isteğinde indirilir)
```

Varsayılan model reposu: **`MuratKeskin0/smart-disaster-hub-ml`**

Farklı bir repo kullanmak için:

```bash
set ML_MODELS_HF_REPO=KULLANICI/repo-adi        # Windows
# export ML_MODELS_HF_REPO=KULLANICI/repo-adi   # Linux/macOS
python scripts/download_models.py
```

---

## Proje yapısı

```
ml-service/
├── README.md                 ← bu dosya
├── SETUP.md                  ← detaylı kurulum ve sorun giderme
├── requirements.txt
├── scripts/
│   ├── download_models.py    ← kullanıcılar: model indir
│   └── upload_models_to_hf.py← maintainer: Hub'a yükle
└── inference/
    ├── app.py                ← FastAPI giriş noktası
    └── services/
        ├── model_assets.py   ← otomatik model indirme
        ├── t2_humanitarian.py
        └── text_analyzer/    ← T1 sınıflandırıcı
```

---

## Maintainer: modelleri Hub'a yükleme

Model ağırlıkları repoya **commit edilmez** (boyut sınırı). İlk kez veya güncelleme için:

```bash
# model.pkl ve vectorizer.pkl dosyalarını şuraya koy:
# inference/services/text_analyzer/models/

huggingface-cli login
python scripts/upload_models_to_hf.py
```

Ayrıntılar: [SETUP.md](./SETUP.md)

---

## Sorun giderme

| Sorun | Çözüm |
|-------|--------|
| `model.pkl not found` | `python scripts/download_models.py` çalıştır; Hub repo public mi kontrol et |
| Servis başlamıyor | `pip install -r requirements.txt` tekrar dene |
| Türkçe çeviri yavaş | İlk istekte MarianMT indirilir (~500 MB), sonrası cache'lenir |
| RoBERTa çalışmıyor | `--roberta` ile indir; yoksa LR varsayılan olarak kullanılır |

Daha fazla detay: [SETUP.md](./SETUP.md)

---

## Lisans ve bağlantılar

- **GitHub:** [muratkeskin0/ml-service](https://github.com/muratkeskin0/ml-service)
- **Model weights:** [MuratKeskin0/smart-disaster-hub-ml](https://huggingface.co/MuratKeskin0/smart-disaster-hub-ml)
- **Ana proje:** Smart Disaster Hub (Spring Boot + Angular + ML Service)
