"""
T1: Disaster Relevance Classification Service
Reddit post'unun afet ile ilgili olup olmadığını tespit eder
Hem Logistic Regression hem de XLM-RoBERTa modellerini destekler
"""
from typing import Tuple, Optional
import pickle
from pathlib import Path


class TextAnalyzer:
    """
    T1: Disaster Relevance Classification için text analyzer
    Otomatik olarak mevcut model tipini algılar:
    - Logistic Regression (model.pkl + vectorizer.pkl)
    - XLM-RoBERTa (Hugging Face format)
    """
    
    def __init__(self, model_path: Optional[str] = None, model_type: Optional[str] = None):
        """
        Text analyzer başlat
        
        Args:
            model_path: Eğitilmiş model klasör yolu (opsiyonel)
                - Logistic Regression: models/ klasörü
                - XLM-RoBERTa: models/xlm_roberta/ klasörü
            model_type: Model tipi ('logistic' veya 'xlm_roberta')
                None ise otomatik algılar
        
        Raises:
            FileNotFoundError: Model dosyaları bulunamazsa
            RuntimeError: Model yüklenirken hata oluşursa
        """
        # Model değişkenleri
        self.model = None
        self.vectorizer = None
        self.tokenizer = None
        self.model_type = None
        
        # Model yolu belirleme
        if model_path:
            model_dir = Path(model_path)
        else:
            # Varsayılan model yolu: services/text_analyzer/models/
            current_file = Path(__file__)
            model_dir = current_file.parent / "models"
        
        # Model tipini algıla veya belirtilen tipi kullan
        if model_type:
            self.model_type = model_type
        else:
            self.model_type = self._detect_model_type(model_dir)
        
        # Model yükleme
        if self.model_type == 'xlm_roberta':
            self._load_xlm_roberta(model_dir)
        else:
            self._load_logistic_regression(model_dir)
    
    def _detect_model_type(self, model_dir: Path) -> str:
        """
        Model tipini otomatik algıla
        
        Args:
            model_dir: Model klasörü
            
        Returns:
            str: 'logistic' veya 'xlm_roberta'
        """
        # XLM-RoBERTa kontrolü (config.json ve pytorch_model.bin veya model.safetensors)
        xlm_roberta_dir = model_dir / "xlm_roberta"
        if xlm_roberta_dir.exists():
            config_file = xlm_roberta_dir / "config.json"
            model_file = xlm_roberta_dir / "pytorch_model.bin"
            safetensors_file = xlm_roberta_dir / "model.safetensors"
            
            if config_file.exists() and (model_file.exists() or safetensors_file.exists()):
                return 'xlm_roberta'
        
        # Alternatif: Direkt xlm_roberta klasörü
        if (model_dir / "config.json").exists():
            model_file = model_dir / "pytorch_model.bin"
            safetensors_file = model_dir / "model.safetensors"
            if model_file.exists() or safetensors_file.exists():
                return 'xlm_roberta'
        
        # Logistic Regression kontrolü (varsayılan)
        return 'logistic'
    
    def _load_logistic_regression(self, model_dir: Path):
        """Logistic Regression model yükle"""
        model_file = model_dir / "model.pkl"
        vectorizer_file = model_dir / "vectorizer.pkl"
        
        if not model_file.exists() or not vectorizer_file.exists():
            raise FileNotFoundError(
                f"Logistic Regression model dosyaları bulunamadı! "
                f"Model: {model_file}, Vectorizer: {vectorizer_file}"
            )
        
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(vectorizer_file, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        print(f"[OK] Logistic Regression Model yüklendi: {model_dir}")
    
    def _load_xlm_roberta(self, model_dir: Path):
        """XLM-RoBERTa model yükle"""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
        except ImportError:
            raise ImportError(
                "XLM-RoBERTa modeli için 'transformers' ve 'torch' paketleri gerekli! "
                "pip install transformers torch"
            )
        
        # Model yolu kontrolü
        xlm_roberta_dir = model_dir / "xlm_roberta"
        if xlm_roberta_dir.exists():
            model_path = xlm_roberta_dir
        elif (model_dir / "config.json").exists():
            model_path = model_dir
        else:
            raise FileNotFoundError(
                f"XLM-RoBERTa model dosyaları bulunamadı! "
                f"Aranan: {xlm_roberta_dir} veya {model_dir}"
            )
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
            self.model.eval()  # Evaluation mode
            print(f"[OK] XLM-RoBERTa Model yüklendi: {model_path}")
        except Exception as e:
            raise RuntimeError(f"XLM-RoBERTa model yüklenirken hata: {e}")
    
    def classify_disaster_relevance(self, text: str) -> Tuple[bool, float]:
        """
        T1: Disaster relevance classification
        
        Post'un afet ile ilgili olup olmadığını tespit eder.
        
        Args:
            text: Reddit post text (title + selftext birleştirilmiş)
            
        Returns:
            Tuple[bool, float]: (is_disaster_related, relevance_score)
                - is_disaster_related: True/False
                - relevance_score: 0.0 - 1.0 arası güven skoru
                
        Raises:
            RuntimeError: Model yüklenmemişse
        """
        if not text or len(text.strip()) == 0:
            return False, 0.0
        
        if not self.model:
            raise RuntimeError("Model yüklenmemiş! Model dosyalarını kontrol edin.")
        
        if self.model_type == 'xlm_roberta':
            return self._classify_with_xlm_roberta(text)
        else:
            return self._classify_with_logistic_regression(text)
    
    def _classify_with_logistic_regression(self, text: str) -> Tuple[bool, float]:
        """
        Logistic Regression ile classification
        
        Args:
            text: Analiz edilecek metin
            
        Returns:
            Tuple[bool, float]: (is_disaster_related, relevance_score)
        """
        if not self.vectorizer:
            raise RuntimeError("Vectorizer yüklenmemiş!")
        
        # TF-IDF vectorization
        text_vectorized = self.vectorizer.transform([text])
        
        # Prediction
        prediction = self.model.predict(text_vectorized)[0]
        probability = self.model.predict_proba(text_vectorized)[0]
        
        # Class 0: not_related, Class 1: disaster_related
        is_related = bool(prediction == 1)
        confidence = float(probability[1])  # Disaster related probability
        
        return is_related, confidence
    
    def _classify_with_xlm_roberta(self, text: str) -> Tuple[bool, float]:
        """
        XLM-RoBERTa ile classification
        
        Args:
            text: Analiz edilecek metin
            
        Returns:
            Tuple[bool, float]: (is_disaster_related, relevance_score)
        """
        import torch
        
        if not self.tokenizer:
            raise RuntimeError("Tokenizer yüklenmemiş!")
        
        # Tokenization
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        # Prediction (no gradient needed)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
        
        # Class 0: not_related, Class 1: disaster_related
        prediction = torch.argmax(probabilities, dim=-1).item()
        confidence = float(probabilities[0][1].item())  # Disaster related probability
        
        is_related = bool(prediction == 1)
        
        return is_related, confidence
