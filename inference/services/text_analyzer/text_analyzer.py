"""
T1: Disaster Relevance Classification Service
Reddit post'unun afet ile ilgili olup olmadığını tespit eder
Logistic Regression modeli kullanır (TF-IDF + Linguistic Features + Character N-grams)
Multi-language support: 100+ dil → İngilizce (HuggingFace MarianMT - ücretsiz)
"""
from typing import Tuple, Optional
import pickle
from pathlib import Path
import numpy as np
from scipy.sparse import hstack, csr_matrix
import logging

logger = logging.getLogger(__name__)

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
    Logistic Regression modeli kullanır (TF-IDF + Linguistic Features + Character N-grams)
    Ücretsiz multi-language translation: HuggingFace MarianMT (100+ dil → İngilizce)
    Desteklenen diller: tr, es, fr, de, ar, ru, ja, ko, zh, hi, vb.
    """
    
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
        
        # Sadece Logistic Regression modeli yükle
        self._load_logistic_regression(model_dir)
        
        # Feature extractor yükleme (linguistic features için)
        self._load_feature_extractor(model_dir)
    
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
    
    def classify_disaster_relevance(self, text: str) -> Tuple[bool, float]:
        """
        T1: Disaster relevance classification
        
        Post'un afet ile ilgili olup olmadığını tespit eder.
        Tüm diller (100+) otomatik olarak İngilizce'ye çevrilir (HuggingFace MarianMT - ücretsiz).
        Desteklenen diller: tr, es, fr, de, ar, ru, ja, ko, zh, hi, vb.
        
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
        
        # Metni sınıflandırma için hazırla (dil algılama ve çeviri)
        text_to_classify = self._prepare_text_for_classification(text)
        
        # Sınıflandırma (Logistic Regression ile)
        return self._classify_with_logistic_regression(text_to_classify)
    
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
    