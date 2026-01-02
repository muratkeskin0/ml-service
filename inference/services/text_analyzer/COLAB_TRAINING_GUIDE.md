# Google Colab ile Model Eğitimi ve Kullanımı

Bu rehber, modeli Google Colab'da eğitip projenizde kullanmanızı sağlar.

## 🎯 Avantajlar

- ✅ **Ücretsiz GPU** (Tesla T4, 15GB RAM)
- ✅ **PC'nizi yormaz** - Eğitim bulutta yapılır
- ✅ **Kolay paylaşım** - Notebook paylaşılabilir
- ✅ **Hızlı eğitim** - GPU ile 2-3 saat

## 📋 Adım 1: Google Colab'a Hazırlık

### 1.1 Notebook'u Aç
1. Google Colab'ı açın: https://colab.research.google.com/
2. `train_xlm_roberta_colab.ipynb` dosyasını yükleyin
3. Veya yeni bir notebook oluşturup kodu kopyalayın

### 1.2 GPU Aktif Et
1. Runtime → Change runtime type
2. Hardware accelerator: **GPU** seçin
3. GPU type: **T4** (varsayılan, ücretsiz)

### 1.3 Dataset'i Yükle

**Yöntem A: Google Drive (Önerilen)**
```python
from google.colab import drive
drive.mount('/content/drive')

# Projenizi Drive'a yükleyin, sonra:
DATASET_PATH = '/content/drive/MyDrive/graduation_project/ml-service/inference/services/text_analyzer/data'
```

**Yöntem B: Zip Dosyası**
1. `data` klasörünü zip'leyin
2. Colab'a yükleyin:
```python
from google.colab import files
uploaded = files.upload()  # data.zip seçin
!unzip data.zip
DATASET_PATH = '/content/data'
```

**Yöntem C: GitHub (Repo public ise)**
```python
!git clone https://github.com/kullaniciadi/graduation_project.git
DATASET_PATH = '/content/graduation_project/ml-service/inference/services/text_analyzer/data'
```

## 📋 Adım 2: Gerekli Dosyaları Yükle

### data_processor.py
`data_processor.py` dosyasını Colab'a yükleyin:

**Yöntem 1: Dosya yükleme**
```python
from google.colab import files
files.upload()  # data_processor.py seçin
```

**Yöntem 2: Drive'dan kopyala**
```python
!cp /content/drive/MyDrive/graduation_project/ml-service/inference/services/text_analyzer/data_processor.py /content/
```

## 📋 Adım 3: Eğitimi Başlat

Notebook'taki tüm hücreleri sırayla çalıştırın:

1. **Paket yükleme** - `!pip install ...`
2. **Dataset yükleme** - Dataset'i bağlayın
3. **Data processor yükleme** - data_processor.py'yi yükleyin
4. **Eğitim kodları** - Sırayla çalıştırın

### Eğitim Süresi
- **GPU ile**: ~2-3 saat (3 epochs)
- **CPU ile**: ~15-20 saat (önerilmez)

## 📋 Adım 4: Modeli İndir

### Yöntem 1: Direkt İndirme (Önerilen)
```python
# Modeli zip'le
!cd /content && zip -r xlm_roberta_model.zip xlm_roberta_model/

# İndir
from google.colab import files
files.download('/content/xlm_roberta_model.zip')
```

### Yöntem 2: Google Drive'a Kaydet
```python
# Drive'a kopyala
!cp -r /content/xlm_roberta_model /content/drive/MyDrive/

# Daha sonra Drive'dan bilgisayarınıza indirin
```

## 📋 Adım 5: Projeye Entegre Et

### 5.1 Model Dosyalarını Yerleştir

İndirdiğiniz `xlm_roberta_model.zip` dosyasını açın ve içeriğini:

```
ml-service/inference/services/text_analyzer/models/xlm_roberta/
├── config.json
├── pytorch_model.bin (veya model.safetensors)
├── tokenizer_config.json
├── tokenizer.json
├── vocab.json
├── merges.txt
└── model_metadata.json
```

### 5.2 Gereken Paketleri Yükle

Projenizde (eğitim yapmadan sadece kullanmak için):

```bash
cd ml-service/inference
pip install transformers torch
```

**Not**: Sadece inference için, eğitim paketleri (`datasets`, `accelerate`) gerekmez.

### 5.3 Test Et

```python
from text_analyzer.text_analyzer import TextAnalyzer

# Otomatik olarak XLM-RoBERTa modelini algılar
analyzer = TextAnalyzer()

# Test
result = analyzer.classify_disaster_relevance("deprem oldu, ev çöktü")
print(result)  # (True, 0.85)
```

## 🔄 Alternatif: Kaggle Notebooks

Kaggle da ücretsiz GPU sunar:

1. https://www.kaggle.com/ adresine gidin
2. New Notebook oluşturun
3. GPU açın (Settings → Accelerator → GPU)
4. Dataset'i yükleyin
5. Aynı kodu kullanın

**Kaggle Avantajları:**
- 30 saat/hafta ücretsiz GPU
- Daha güçlü GPU'lar (P100, V100)

## 💾 Model Dosya Boyutları

Eğitilmiş XLM-RoBERTa modeli yaklaşık:
- **Base model**: ~500 MB
- **Large model**: ~1.5 GB

Zip'lenmiş: ~200-500 MB

## ⚡ Inference Hızı

Eğitilmiş model inference hızı:
- **CPU**: ~100-200ms/metin
- **GPU**: ~10-20ms/metin

API için yeterince hızlı!

## 🐛 Sorun Giderme

### "CUDA out of memory"
**Çözüm**: Batch size'ı küçültün (16 → 8)

### "Model dosyaları bulunamadı"
**Çözüm**: 
- Model yolunu kontrol edin
- `models/xlm_roberta/` klasörünün varlığını kontrol edin

### "transformers modülü bulunamadı"
**Çözüm**:
```bash
pip install transformers torch
```

### Colab'da eğitim çok yavaş
**Çözüm**: 
- GPU'nun aktif olduğunu kontrol edin
- Runtime → Change runtime type → GPU

## 📊 Beklenen Performans

Eğitilmiş XLM-RoBERTa modeli ile:

- **Test Accuracy**: %75-85 (Logistic Regression: %60)
- **Real Disasters**: %70-80 (önceki: %40)
- **Spelling Errors**: %60-70 (önceki: %20)
- **Ambiguous Context**: %70-75 (önceki: %33)

## 🎓 Sonraki Adımlar

1. ✅ Modeli Colab'da eğitin
2. ✅ Modeli indirin ve projeye ekleyin
3. ✅ `text_analyzer.py` otomatik algılar
4. ✅ API'yi test edin
5. ✅ Production'a deploy edin

## 💡 İpuçları

1. **Colab'da eğitim sırasında**: Notebook'u kapatmayın, oturum kapanabilir
2. **Drive kullanın**: Modeli Drive'a kaydedin, güvenli
3. **Checkpoint'leri kontrol edin**: En iyi model otomatik seçilir
4. **Test edin**: İndirdikten sonra mutlaka test edin

Herhangi bir sorun yaşarsanız, hata mesajlarını paylaşın!




