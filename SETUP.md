# ML Service — Kurulum (ücretsiz model hosting)

Genel proje tanıtımı ve API örnekleri için: [README.md](./README.md)

Bu servis **kodu GitHub'da**, **ağırlıkları Hugging Face Hub'da** (ücretsiz) tutar. Clone eden biri birkaç komutla çalıştırabilir.

## 1. Hugging Face hesabı (ücretsiz)

1. [https://huggingface.co/join](https://huggingface.co/join) — ücretsiz kayıt
2. Terminalde giriş:

```bash
pip install huggingface_hub
huggingface-cli login
```

Public model repo'ları **ücretsiz** ve sınırsız indirilebilir.

## 2. Modelleri Hub'a yükle (sadece ilk kez — senin yapman gereken)

Eğitilmiş dosyaların şu klasörde olmalı:

```
inference/services/text_analyzer/models/
├── model.pkl              ← zorunlu
├── vectorizer.pkl         ← zorunlu
├── char_vectorizer.pkl    ← opsiyonel
├── feature_extractor.pkl  ← opsiyonel
└── roberta/               ← opsiyonel (use_roberta=true için)
```

Yükleme:

```bash
cd ml-service
pip install huggingface_hub
huggingface-cli login
python scripts/upload_models_to_hf.py
```

Varsayılan repo: `muratkeskin0/smart-disaster-hub-ml`  
Farklı isim için: `python scripts/upload_models_to_hf.py --repo KULLANICI_ADIN/repo-adi`

Hub'da public repo oluşur: `https://huggingface.co/muratkeskin0/smart-disaster-hub-ml`

## 3. Başkasının (veya yeni makinenin) kurulumu

```bash
git clone https://github.com/muratkeskin0/ml-service.git
cd ml-service
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS
pip install -r requirements.txt
python scripts/download_models.py
```

FastText dil modeli Meta CDN'den otomatik iner (~1 MB, `.ftz`).  
Logistic Regression dosyaları Hugging Face'den iner.

Servisi başlat:

```bash
cd inference
python app.py
# veya: uvicorn app:app --host 0.0.0.0 --port 8000
```

Health check: [http://localhost:8000/health](http://localhost:8000/health)

## 4. Opsiyonel: RoBERTa

```bash
python scripts/download_models.py --roberta
```

RoBERTa ağırlıkları Hub'da yoksa servis yine de Logistic Regression ile çalışır.

## 5. Ortam değişkenleri

| Değişken | Açıklama |
|----------|----------|
| `ML_MODELS_HF_REPO` | Model repo id (varsayılan: `muratkeskin0/smart-disaster-hub-ml`) |

## 6. Sorun giderme

| Hata | Çözüm |
|------|--------|
| `model.pkl not found` | Hub'a henüz yüklenmemiş — `upload_models_to_hf.py` çalıştır |
| `401 Unauthorized` (upload) | `huggingface-cli login` |
| `Repository Not Found` (download) | Repo public mi? URL doğru mu? |
| FastText indirilemedi | İnternet gerekli; yoksa heuristik dil algılama kullanılır |
| İlk çeviri yavaş | MarianMT ilk seferde Hugging Face'den indirilir (~500 MB) |

## Neden Hugging Face?

- ML/NLP projelerinde yaygın, **ücretsiz public hosting**
- `transformers` ile uyumlu
- Git repo'sunu GB'larla şişirmez
- Mezuniyet / portfolyo projelerinde standart pratik
