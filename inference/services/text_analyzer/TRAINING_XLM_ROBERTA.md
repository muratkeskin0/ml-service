# XLM-RoBERTa Model Eğitimi Rehberi

Bu doküman, XLM-RoBERTa modeli ile disaster relevance classification modeli eğitmek için rehberdir.

## 📋 Gereksinimler

### 1. Python Paketleri

```bash
cd ml-service/inference
pip install -r requirements.txt
```

Yüklenmesi gereken paketler:
- `torch>=2.0.0` - PyTorch
- `transformers>=4.35.0` - Hugging Face Transformers
- `datasets>=2.14.0` - Hugging Face Datasets
- `accelerate>=0.24.0` - Hugging Face Accelerate
- `scikit-learn>=1.3.0` - Sklearn (mevcut)
- `numpy>=1.24.0` - NumPy (mevcut)

### 2. GPU (Önerilen)

XLM-RoBERTa eğitimi CPU'da çok yavaş olabilir. GPU kullanılması önerilir:

- **NVIDIA GPU** (CUDA desteği)
- En az **8GB GPU Memory** (batch_size=16 için)
- **CUDA 11.8+** ve **cuDNN**

GPU kontrolü:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### 3. Disk Alanı

- Model dosyaları: ~500 MB - 1 GB (xlm-roberta-base)
- Eğitim sırasında checkpoint'ler: ~2-3 GB ekstra
- Toplam önerilen: **5 GB+ boş alan**

## 🚀 Eğitim Adımları

### Adım 1: Veri Kontrolü

Eğitim öncesi veri setlerinin mevcut olduğundan emin olun:

```bash
cd ml-service/inference/services/text_analyzer
python -c "from data_processor import DataProcessor; dp = DataProcessor(); texts, labels, sources = dp.process_all_datasets(); print(f'Total samples: {len(texts)}')"
```

### Adım 2: Model Eğitimi

**Basit kullanım (varsayılan parametreler):**
```bash
cd ml-service/inference/services/text_analyzer
python train_xlm_roberta.py
```

**Parametreleri özelleştirme:**

Script içinde `if __name__ == "__main__":` bölümünde parametreleri değiştirebilirsiniz:

```python
train_xlm_roberta(
    model_name="xlm-roberta-base",  # veya "xlm-roberta-large"
    max_length=512,                  # Maximum sequence length
    batch_size=16,                   # GPU varsa 16, yoksa 4
    learning_rate=2e-5,              # Learning rate
    num_epochs=3,                    # Epoch sayısı
    use_gpu=True                     # GPU kullanımı
)
```

### Adım 3: Eğitim Süreci

Eğitim sırasında şunları göreceksiniz:

1. **Dataset yükleme** - Tüm veri setleri birleştirilir
2. **Train/Val/Test split** - Veriler %80/%10/%10 olarak bölünür
3. **Model yükleme** - XLM-RoBERTa base model indirilir
4. **Tokenization** - Metinler tokenize edilir
5. **Eğitim** - Her epoch'ta loss ve metrics gösterilir
6. **Test evaluation** - Final test seti üzerinde değerlendirme

Örnek çıktı:
```
[1/7] Datasetler yukleniyor...
  - Toplam kayit: 191761
  - Disaster related: 117529 (61.3%)
  - Not related: 74232 (38.7%)

[2/7] Train-test split yapiliyor...
  - Train set: 138068 kayit
  - Validation set: 15341 kayit
  - Test set: 38352 kayit

[3/7] Tokenizer yukleniyor: xlm-roberta-base
[4/7] Model yukleniyor: xlm-roberta-base

[7/7] Model egitimi basliyor...
Epoch 1/3: 100%|████████| 8629/8629 [45:23<00:00, loss=0.234, val_loss=0.156]
Epoch 2/3: 100%|████████| 8629/8629 [45:15<00:00, loss=0.112, val_loss=0.098]
Epoch 3/3: 100%|████████| 8629/8629 [45:20<00:00, loss=0.087, val_loss=0.092]
```

## ⚙️ Parametre Ayarı (Hyperparameter Tuning)

### Batch Size
- **GPU (8GB+)**: 16
- **GPU (4-8GB)**: 8
- **CPU**: 2-4

### Learning Rate
- **Önerilen**: 2e-5 (2x10^-5)
- **Küçük dataset**: 3e-5
- **Büyük dataset**: 1e-5

### Max Length
- **Kısa metinler (tweetler)**: 256
- **Orta metinler**: 512 (varsayılan)
- **Uzun metinler**: 768 (daha fazla GPU memory gerektirir)

### Num Epochs
- **Önerilen**: 3-5
- Early stopping ile otomatik durur (patience=2)

## 📊 Beklenen Performans

XLM-RoBERTa base model ile beklenen performans:

- **Test Accuracy**: %75-85 (mevcut Logistic Regression: %60)
- **Test F1-Score**: %75-85
- **Eğitim süresi** (GPU ile): ~2-3 saat (3 epochs)
- **Eğitim süresi** (CPU ile): ~15-20 saat (önerilmez)

## 📁 Çıktı Dosyaları

Eğitim tamamlandığında `models/xlm_roberta/` klasöründe:

```
xlm_roberta/
├── config.json              # Model konfigürasyonu
├── pytorch_model.bin        # Model weights
├── tokenizer_config.json    # Tokenizer config
├── tokenizer.json           # Tokenizer
├── vocab.json               # Vocabulary
├── merges.txt               # BPE merges
├── model_metadata.json      # Eğitim metadata'sı
└── logs/                    # Training logs
```

## 🔄 Model Kullanımı

Eğitilmiş modeli kullanmak için `text_analyzer.py` dosyasını güncellemeniz gerekir.

Alternatif: Eğitilmiş modeli direkt test etmek:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

model_path = "models/xlm_roberta"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Test
result = classifier("deprem oldu, ev çöktü, yardım lazım")
print(result)
```

## ⚠️ Sorun Giderme

### CUDA Out of Memory
**Sorun**: GPU memory yetersiz
**Çözüm**:
- `batch_size`'ı küçült (16 → 8 → 4)
- `max_length`'i küçült (512 → 256)
- Gradient accumulation kullan

### Eğitim çok yavaş
**Sorun**: CPU kullanılıyor veya batch size çok küçük
**Çözüm**:
- GPU kontrolü yapın
- Batch size'ı artırın (GPU memory izin veriyorsa)
- `fp16=True` kullanın (mixed precision)

### Model indirme hatası
**Sorun**: İnternet bağlantısı veya Hugging Face erişimi
**Çözüm**:
```bash
# Hugging Face token gerekebilir
huggingface-cli login
```

### Dataset yükleme hatası
**Sorun**: Veri dosyaları eksik veya bozuk
**Çözüm**:
- `data/` klasöründeki dosyaları kontrol edin
- `data_processor.py` ile test edin

## 📚 Ek Kaynaklar

- [XLM-RoBERTa Paper](https://arxiv.org/abs/1911.02116)
- [Hugging Face Transformers Docs](https://huggingface.co/docs/transformers)
- [XLM-RoBERTa Model Card](https://huggingface.co/xlm-roberta-base)

## 💡 İpuçları

1. **İlk eğitimde küçük batch size kullanın** (GPU memory test için)
2. **Validation loss'u izleyin** - Overfitting olup olmadığını gösterir
3. **Early stopping aktif** - Gereksiz epoch'ları durdurur
4. **Checkpoint'ler kaydedilir** - En iyi model otomatik seçilir
5. **Logs klasörünü kontrol edin** - TensorBoard ile görselleştirebilirsiniz

## 🔄 Mevcut Logistic Regression ile Karşılaştırma

| Özellik | Logistic Regression | XLM-RoBERTa |
|---------|---------------------|-------------|
| Accuracy | %60 | %75-85 (beklenen) |
| Bağlam Anlama | Zayıf | Güçlü |
| Yazım Hatası Toleransı | Düşük | Yüksek |
| Eğitim Süresi | ~5-10 dakika | ~2-3 saat (GPU) |
| Inference Hızı | Çok Hızlı | Hızlı |
| Model Boyutu | ~50 MB | ~500 MB |
| GPU Gereksinimi | Hayır | Önerilir |




