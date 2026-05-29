"""
T1: Disaster Relevance Classification Service
Reddit post'unun afet ile ilgili olup olmadığını tespit eder
Dual-model: Logistic Regression (varsayılan) + RoBERTa (opsiyonel, use_roberta=True)
Multi-language support: 100+ dil → İngilizce (HuggingFace MarianMT - ücretsiz)
"""
from typing import Tuple, Optional, Dict, Any
import json
import pickle
from pathlib import Path
import numpy as np
from scipy.sparse import hstack, csr_matrix
import logging

logger = logging.getLogger(__name__)

# RoBERTa (lazy import - transformers/torch gerekir)
ROBERTA_AVAILABLE = False
try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    ROBERTA_AVAILABLE = True
except ImportError:
    pass

try:
    from .feature_extractor import FeatureExtractor
except ImportError:
    try:
        from feature_extractor import FeatureExtractor
    except ImportError:
        # Fallback if feature_extractor not available
        FeatureExtractor = None

# Ücretsiz translation servisi (HuggingFace MarianMT)
# Google API kullanılmıyor, tamamen ücretsiz ve offline çalışır
try:
    from .translation_service_free import FreeTranslationService
    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False
    logger.warning("Ücretsiz translation servisi bulunamadı. Yüklemek için: pip install transformers torch")


class TextAnalyzer:
    """
    T1: Disaster Relevance Classification için text analyzer
    Dual-model: Logistic Regression (hızlı) + RoBERTa (opsiyonel, daha yüksek doğruluk)
    Ücretsiz multi-language translation: HuggingFace MarianMT (100+ dil → İngilizce)
    Desteklenen diller: tr, es, fr, de, ar, ru, ja, ko, zh, hi, vb.
    """
    
    # RoBERTa model alt klasör adı
    ROBERTA_SUBDIR = "roberta"
    
    def __init__(
        self, 
        model_path: Optional[str] = None, 
        model_type: Optional[str] = None,
        enable_translation: bool = True  # Varsayılan: Aktif (HuggingFace MarianMT - ücretsiz)
    ):
        """
        Text analyzer başlat
        
        Args:
            model_path: Eğitilmiş model klasör yolu (opsiyonel)
                - Logistic Regression: models/ klasörü
                - RoBERTa: models/roberta/ (fine-tuned RoBERTaForSequenceClassification)
            enable_translation: Çeviri özelliğini aktif et
                True ise HuggingFace MarianMT (ücretsiz) ile Türkçe metinler çevrilir
                False ise sadece İngilizce metinler desteklenir
        
        Raises:
            FileNotFoundError: Model dosyaları bulunamazsa
            RuntimeError: Model yüklenirken hata oluşursa
        """
        # Model değişkenleri
        self.model = None
        self.vectorizer = None
        self.char_vectorizer = None
        self.feature_extractor = None
        self.use_linguistic_features = False
        self.use_character_ngrams = False
        
        # RoBERTa (lazy load)
        self._roberta_model = None
        self._roberta_tokenizer = None
        self._roberta_is_multitask = False
        self._model_dir = None
        
        # Language detection ve Free Translation (HuggingFace MarianMT)
        # FastText modeli mevcut: services/language_detector/models/lid.176.bin
        self.enable_translation = enable_translation
        self.language_detector = None
        self.translation_service = None
        
        # Language detector yükle
        try:
            from .language_detector import LanguageDetector
            self.language_detector = LanguageDetector()
            if self.language_detector.use_fasttext:
                logger.info("FastText language detector aktif")
            else:
                logger.info("Basit heuristik language detector aktif (FastText yok)")
        except Exception as e:
            logger.warning(f"Language detector yüklenemedi: {e}")
            self.language_detector = None
        
        # Free translation service yükle (HuggingFace MarianMT - tamamen ücretsiz)
        if enable_translation:
            try:
                from .translation_service_free import FreeTranslationService
                self.translation_service = FreeTranslationService()
                logger.info("Ücretsiz translation servisi yüklendi (HuggingFace MarianMT)")
            except Exception as e:
                logger.warning(f"Translation servisi yüklenemedi: {e}. Çeviri özelliği devre dışı.")
                logger.warning("Yüklemek için: pip install transformers torch")
                self.translation_service = None
        else:
            logger.info("Translation servisi kapalı - sadece İngilizce metinler destekleniyor")
        
        # Model yolu belirleme
        if model_path:
            model_dir = Path(model_path)
        else:
            # Varsayılan model yolu: services/text_analyzer/models/
            current_file = Path(__file__)
            model_dir = current_file.parent / "models"
        self._model_dir = model_dir
        
        # Logistic Regression modeli yükle
        self._load_logistic_regression(model_dir)
        
        # Feature extractor yükleme (linguistic features için)
        self._load_feature_extractor(model_dir)
        
        # RoBERTa: lazy load (models/roberta/ varsa ilk use_roberta=True'da yüklenecek)
        roberta_dir = model_dir / self.ROBERTA_SUBDIR
        if roberta_dir.exists() and (roberta_dir / "config.json").exists():
            logger.info("RoBERTa model dizini mevcut (lazy load): %s", roberta_dir)
        elif ROBERTA_AVAILABLE:
            logger.info("RoBERTa model dizini yok; use_roberta istekleri Logistic Regression kullanacak.")
    
    def _load_logistic_regression(self, model_dir: Path):
        """Logistic Regression model yükle"""
        try:
            from ..model_assets import ensure_logistic_regression_models
        except ImportError:
            from model_assets import ensure_logistic_regression_models

        ensure_logistic_regression_models(model_dir)

        model_file = model_dir / "model.pkl"
        vectorizer_file = model_dir / "vectorizer.pkl"
        
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(vectorizer_file, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        # Character-level vectorizer yükleme (opsiyonel)
        char_vectorizer_file = model_dir / "char_vectorizer.pkl"
        if char_vectorizer_file.exists():
            try:
                with open(char_vectorizer_file, 'rb') as f:
                    self.char_vectorizer = pickle.load(f)
                self.use_character_ngrams = True
                print(f"[OK] Character-level vectorizer yüklendi: {char_vectorizer_file}")
            except Exception as e:
                print(f"[WARNING] Character-level vectorizer yüklenemedi: {e}")
                self.char_vectorizer = None
                self.use_character_ngrams = False
        else:
            self.char_vectorizer = None
            self.use_character_ngrams = False
        
        print(f"[OK] Logistic Regression Model yüklendi: {model_dir}")
    
    def _load_feature_extractor(self, model_dir: Path):
        """Feature extractor yükle (linguistic features için)"""
        feature_extractor_file = model_dir / "feature_extractor.pkl"
        
        if feature_extractor_file.exists():
            try:
                with open(feature_extractor_file, 'rb') as f:
                    self.feature_extractor = pickle.load(f)
                self.use_linguistic_features = True
                print(f"[OK] Feature Extractor yüklendi: {feature_extractor_file}")
            except Exception as e:
                print(f"[WARNING] Feature extractor yüklenemedi: {e}")
                self.feature_extractor = None
                self.use_linguistic_features = False
        else:
            # Try to create new feature extractor
            if FeatureExtractor:
                try:
                    self.feature_extractor = FeatureExtractor()
                    self.use_linguistic_features = True
                    print(f"[INFO] Yeni Feature Extractor oluşturuldu")
                except Exception as e:
                    print(f"[WARNING] Feature extractor oluşturulamadı: {e}")
                    self.feature_extractor = None
                    self.use_linguistic_features = False
            else:
                print(f"[INFO] Feature extractor kullanılmayacak (linguistic features yok)")
                self.feature_extractor = None
                self.use_linguistic_features = False
    
    def _is_multitask_roberta(self) -> bool:
        """models/roberta/ config'inde multitask var mı?"""
        if self._model_dir is None:
            return False
        cfg = self._model_dir / self.ROBERTA_SUBDIR / "config.json"
        if not cfg.exists():
            return False
        try:
            with open(cfg, "r", encoding="utf-8") as f:
                return json.load(f).get("multitask", False)
        except Exception:
            return False

    def _load_roberta(self) -> bool:
        """RoBERTa modelini lazy-load et. Multi-task (T1+T2) veya sadece T1."""
        if self._roberta_model is not None and self._roberta_tokenizer is not None:
            return True
        if not ROBERTA_AVAILABLE or self._model_dir is None:
            return False
        roberta_dir = self._model_dir / self.ROBERTA_SUBDIR
        if not roberta_dir.exists() or not (roberta_dir / "config.json").exists():
            return False
        try:
            self._roberta_tokenizer = AutoTokenizer.from_pretrained(str(roberta_dir))
            if self._is_multitask_roberta():
                try:
                    from .model_multitask_roberta import MultiTaskRobertaForClassification
                except ImportError:
                    from model_multitask_roberta import MultiTaskRobertaForClassification
                # Multi-task model mimarisini doğrudan bu klasördeki config + ağırlıklarla kur
                # (roberta-base stringi yerine yerel klasör yolunu veriyoruz)
                self._roberta_model = MultiTaskRobertaForClassification(base_model_name=str(roberta_dir))
                state = torch.load(roberta_dir / "pytorch_model.bin", map_location="cpu")
                # Yapı uyumsuzluklarına karşı strict=False (eksik/ekstra key'ler tolere edilir)
                self._roberta_model.load_state_dict(state, strict=False)
                self._roberta_model.eval()
                self._roberta_is_multitask = True
                logger.info("RoBERTa multi-task (T1+T2) modeli yüklendi: %s", roberta_dir)
            else:
                self._roberta_model = AutoModelForSequenceClassification.from_pretrained(str(roberta_dir))
                self._roberta_model.eval()
                logger.info("RoBERTa modeli yüklendi: %s", roberta_dir)
            return True
        except Exception as e:
            logger.warning("RoBERTa yüklenemedi: %s", e)
            return False
    
    def classify_disaster_relevance(
        self, text: str, use_roberta: bool = False, return_t2: bool = False
    ) -> Tuple[bool, float, str, Optional[Dict[str, Any]]]:
        """
        T1: Disaster relevance classification. return_t2=True ve multi-task RoBERTa ise T2 de aynı forward'dan döner.
        
        Returns:
            (is_disaster_related, relevance_score, model_used, t2_dict or None)
        """
        if not text or len(text.strip()) == 0:
            return False, 0.0, "logistic_regression", None
        
        if not self.model:
            raise RuntimeError("Model yüklenmemiş! Model dosyalarını kontrol edin.")
        
        text_to_classify = self._prepare_text_for_classification(text)
        t2_result = None

        if use_roberta and self._load_roberta():
            if return_t2 and self._roberta_is_multitask:
                is_related, score, t2_result = self._classify_with_roberta_multitask(text_to_classify)
            else:
                is_related, score = self._classify_with_roberta(text_to_classify)
            return is_related, score, "roberta", t2_result
        is_related, score = self._classify_with_logistic_regression(text_to_classify)
        return is_related, score, "logistic_regression", None
    
    def _prepare_text_for_classification(self, text: str) -> str:
        """
        Metni sınıflandırma için hazırla (dil algılama ve çeviri)
        
        Args:
            text: Orijinal metin

        Returns:
            str: Sınıflandırma için hazır metin (İngilizce'ye çevrilmiş veya orijinal)
        
        Not: Multi-language support - 100+ dil desteklenir
            Tüm diller İngilizce'ye çevrilir (tr, es, fr, de, ar, ru, ja, ko, vb.)
        """
        # Dil algılama ve çeviri (eğer aktifse)
        if self.enable_translation and self.language_detector and self.translation_service:
            try:
                # Dil algılama (176 dil desteği)
                detected_lang = self.language_detector.detect(text)
                
                # İngilizce değilse çevir (multi-language model ile)
                if detected_lang != 'en':
                    translated = self.translation_service.translate_to_english(text, source_lang=detected_lang)
                    logger.debug(f"{detected_lang.upper()} metin çevrildi: '{text[:50]}...' -> '{translated[:50]}...'")
                    return translated
                else:
                    # Zaten İngilizce - çeviri yok
                    return text
            except Exception as e:
                logger.warning(f"Çeviri hatası: {e}, orijinal metin kullanılıyor")
                return text
        else:
            # Translation kapalı - sadece orijinal metni kullan
            return text
    
    def _classify_with_roberta(self, text: str) -> Tuple[bool, float]:
        """
        RoBERTa (HuggingFace) ile binary classification.
        Label 1 = disaster_related, Label 0 = not_related.
        """
        if self._roberta_model is None or self._roberta_tokenizer is None:
            raise RuntimeError("RoBERTa modeli yüklenmemiş.")
        max_length = 512
        inputs = self._roberta_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        with torch.no_grad():
            out = self._roberta_model(**inputs)
            logits = out.logits if not self._roberta_is_multitask else out.logits[0]
        probs = torch.softmax(logits, dim=1)
        prob_disaster = float(probs[0][1].item())
        return prob_disaster >= 0.5, prob_disaster

    def _classify_with_roberta_multitask(self, text: str) -> Tuple[bool, float, Dict[str, Any]]:
        """Multi-task RoBERTa: tek forward ile T1 + T2. (is_related, score, t2_dict) döner."""
        if self._roberta_model is None or self._roberta_tokenizer is None:
            raise RuntimeError("RoBERTa modeli yüklenmemiş.")
        from .model_multitask_roberta import CATEGORY_NAMES
        max_length = 512
        inputs = self._roberta_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        with torch.no_grad():
            out = self._roberta_model(**inputs)
        logits_d, logits_h, logits_c = out.logits
        probs_d = torch.softmax(logits_d, dim=1)
        probs_h = torch.softmax(logits_h, dim=1)
        probs_c = torch.softmax(logits_c, dim=1)
        prob_disaster = float(probs_d[0][1].item())
        is_related = prob_disaster >= 0.5
        prob_help = float(probs_h[0][1].item())
        is_help = prob_help >= 0.5
        cat_probs = {CATEGORY_NAMES[i]: float(probs_c[0][i].item()) for i in range(len(CATEGORY_NAMES))}
        threshold = 0.2
        labels = [CATEGORY_NAMES[i] for i in range(len(CATEGORY_NAMES)) if cat_probs[CATEGORY_NAMES[i]] >= threshold]
        if not labels:
            labels = [CATEGORY_NAMES[probs_c[0].argmax().item()]]
        t2_dict = {
            "is_help_request": is_help,
            "help_request_probability": round(prob_help, 4),
            "humanitarian_labels": labels,
            "category_probabilities": {k: round(v, 4) for k, v in cat_probs.items()},
        }
        return is_related, prob_disaster, t2_dict
    
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
        
        # TF-IDF vectorization (word-level)
        text_vectorized = self.vectorizer.transform([text])
        
        # Character-level n-grams (if available)
        if self.use_character_ngrams and self.char_vectorizer:
            try:
                text_char_vectorized = self.char_vectorizer.transform([text])
                # Combine word-level and character-level
                text_vectorized = hstack([text_vectorized, text_char_vectorized])
            except Exception as e:
                print(f"[WARNING] Character n-grams eklenemedi: {e}, sadece word-level kullanılıyor")
        
        # Linguistic features (if available)
        if self.use_linguistic_features and self.feature_extractor:
            try:
                linguistic_features = self.feature_extractor.extract_linguistic_features(text)
                feature_names = self.feature_extractor.get_feature_names()[:17]  # Only linguistic (17 features)
                linguistic_array = np.array([[linguistic_features[key] for key in feature_names]])
                linguistic_sparse = csr_matrix(linguistic_array)
                
                # Combine TF-IDF and linguistic features
                combined_features = hstack([text_vectorized, linguistic_sparse])
            except Exception as e:
                print(f"[WARNING] Linguistic features eklenemedi: {e}, sadece TF-IDF kullanılıyor")
                # Model linguistic features bekliyorsa, sıfırlarla doldur
                if self.model.n_features_in_ == text_vectorized.shape[1] + 17:
                    # 17 sıfır feature ekle
                    zeros = csr_matrix(np.zeros((1, 17)))
                    combined_features = hstack([text_vectorized, zeros])
                else:
                    combined_features = text_vectorized
        else:
            # Linguistic features yok ama model bekliyorsa, sıfırlarla doldur
            if hasattr(self.model, 'n_features_in_') and self.model.n_features_in_ == text_vectorized.shape[1] + 17:
                zeros = csr_matrix(np.zeros((1, 17)))
                combined_features = hstack([text_vectorized, zeros])
            else:
                combined_features = text_vectorized
        
        # Prediction
        prediction = self.model.predict(combined_features)[0]
        probability = self.model.predict_proba(combined_features)[0]
        
        # Class 0: not_related, Class 1: disaster_related
        is_related = bool(prediction == 1)
        confidence = float(probability[1])  # Disaster related probability
        
        return is_related, confidence
    