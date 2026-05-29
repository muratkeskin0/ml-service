"""
Language Detector
FastText modeli ile dil algılama (öncelikli)
Fallback: Basit heuristik (FastText yoksa)
"""
import re
from typing import Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# FastText import (opsiyonel)
try:
    import fasttext
    FASTTEXT_AVAILABLE = True
except ImportError:
    FASTTEXT_AVAILABLE = False
    logger.warning("fasttext modülü bulunamadı. Basit heuristik kullanılacak.")


class LanguageDetector:
    """
    Dil algılama servisi
    Öncelikle FastText modeli kullanır, yoksa basit heuristik kullanır
    """
    
    # Türkçe karakterler (fallback için)
    TURKISH_CHARS = set('çğıöşüÇĞİÖŞÜ')
    
    # Türkçe yaygın kelimeler (fallback için)
    TURKISH_COMMON_WORDS = {
        've', 'bir', 'bu', 'ile', 'için', 'olan', 'var', 'yok',
        'deprem', 'sel', 'yangın', 'yardım', 'acil', 'imdat', 'kurtarma'
    }
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Language detector başlat
        
        Args:
            model_path: FastText model yolu (opsiyonel)
                None ise otomatik bulur: services/language_detector/models/lid.176.bin
        """
        self.fasttext_model = None
        self.use_fasttext = False
        
        # FastText model yolu belirleme
        if model_path:
            model_file = Path(model_path)
        else:
            # Otomatik model yolu bulma
            current_file = Path(__file__)
            # services/text_analyzer/language_detector.py -> services/language_detector/models/
            services_dir = current_file.parent.parent
            models_dir = services_dir / "language_detector" / "models"
            try:
                from ..model_assets import ensure_fasttext_model
            except ImportError:
                from model_assets import ensure_fasttext_model
            ensure_fasttext_model(models_dir)
            model_file = models_dir / "lid.176.bin"
            if not model_file.exists():
                model_file = models_dir / "lid.176.ftz"
        
        # FastText model yükleme
        if FASTTEXT_AVAILABLE and model_file.exists():
            try:
                self.fasttext_model = fasttext.load_model(str(model_file))
                self.use_fasttext = True
                logger.info(f"FastText language detector yüklendi: {model_file}")
            except Exception as e:
                logger.warning(f"FastText model yüklenemedi: {e}, basit heuristik kullanılacak")
                self.fasttext_model = None
                self.use_fasttext = False
        else:
            if not FASTTEXT_AVAILABLE:
                logger.warning("fasttext modülü yok, basit heuristik kullanılacak")
            elif not model_file.exists():
                logger.warning(f"FastText model bulunamadı: {model_file}, basit heuristik kullanılacak")
            self.use_fasttext = False
    
    def detect(self, text: str) -> str:
        """
        Metnin dilini algıla
        
        Args:
            text: Algılanacak metin
        
        Returns:
            str: ISO 639-1 dil kodu ('tr', 'en', 'es', 'fr', 'de', 'ar', 'ru', vb.)
                FastText 176 dil destekler
        """
        if not text or len(text.strip()) == 0:
            return 'en'  # Varsayılan İngilizce
        
        # FastText kullan (varsa)
        if self.use_fasttext and self.fasttext_model:
            return self._detect_with_fasttext(text)
        else:
            # Fallback: Basit heuristik
            return self._detect_with_heuristic(text)
    
    def _detect_with_fasttext(self, text: str) -> str:
        """
        FastText ile dil algılama (176 dil desteği)
        
        Args:
            text: Algılanacak metin
        
        Returns:
            str: ISO 639-1 dil kodu ('tr', 'en', 'es', 'fr', 'de', 'ar', 'ru', vb.)
        """
        try:
            # FastText prediction: (('__label__tr',), array([0.9999]))
            predictions = self.fasttext_model.predict(text, k=1)
            label = predictions[0][0]  # '__label__tr', '__label__es', vb.
            
            # Label formatı: '__label__tr' -> 'tr'
            # FastText label formatı: '__label__<lang_code>'
            if label.startswith('__label__'):
                lang_code = label.replace('__label__', '').lower()
                logger.debug(f"FastText dil algıladı: {lang_code}")
                return lang_code
            else:
                # Fallback: Label'den dil kodunu çıkar
                lang_code = label.lower().replace('__label__', '')
                if len(lang_code) == 2:  # ISO 639-1 formatı
                    return lang_code
                else:
                    logger.debug(f"FastText farklı format algıladı: {label}, varsayılan İngilizce")
                    return 'en'
        
        except Exception as e:
            logger.warning(f"FastText detection hatası: {e}, fallback kullanılıyor")
            return self._detect_with_heuristic(text)
    
    def _detect_with_heuristic(self, text: str) -> str:
        """
        Basit heuristik ile dil algılama (fallback)
        
        Args:
            text: Algılanacak metin
        
        Returns:
            str: 'tr' veya 'en'
        """
        text_lower = text.lower()
        
        # 1. Türkçe karakter kontrolü
        has_turkish_chars = any(char in self.TURKISH_CHARS for char in text)
        
        # 2. Türkçe yaygın kelime kontrolü
        words = set(re.findall(r'\b\w+\b', text_lower))
        turkish_word_count = len(words.intersection(self.TURKISH_COMMON_WORDS))
        
        # 3. Karar: Türkçe karakter varsa veya yeterli Türkçe kelime varsa
        if has_turkish_chars:
            return 'tr'
        
        if turkish_word_count >= 2:  # En az 2 Türkçe kelime
            return 'tr'
        
        # Varsayılan: İngilizce
        return 'en'
    
    def detect_with_confidence(self, text: str) -> Tuple[str, float]:
        """
        Metnin dilini algıla ve güven skoru ver
        
        Args:
            text: Algılanacak metin
        
        Returns:
            tuple: (dil, güven_skoru)
                - dil: ISO 639-1 dil kodu ('tr', 'en', 'es', 'fr', vb.)
                - güven_skoru: 0.0-1.0
        """
        if not text or len(text.strip()) == 0:
            return 'en', 0.5
        
        # FastText kullan (varsa)
        if self.use_fasttext and self.fasttext_model:
            return self._detect_with_fasttext_confidence(text)
        else:
            # Fallback: Basit heuristik
            return self._detect_with_heuristic_confidence(text)
    
    def _detect_with_fasttext_confidence(self, text: str) -> Tuple[str, float]:
        """
        FastText ile dil algılama ve güven skoru (176 dil desteği)
        
        Args:
            text: Algılanacak metin
        
        Returns:
            tuple: (dil, güven_skoru)
                - dil: ISO 639-1 dil kodu
                - güven_skoru: 0.0-1.0
        """
        try:
            # FastText prediction: (('__label__tr',), array([0.9999]))
            predictions = self.fasttext_model.predict(text, k=1)
            label = predictions[0][0]  # '__label__tr', '__label__es', vb.
            confidence = float(predictions[1][0])  # 0.9999
            
            # Label formatı: '__label__<lang_code>' -> '<lang_code>'
            if label.startswith('__label__'):
                lang_code = label.replace('__label__', '').lower()
                logger.debug(f"FastText dil algıladı: {lang_code} (güven: {confidence:.2f})")
                return lang_code, confidence
            else:
                # Fallback
                lang_code = label.lower().replace('__label__', '')
                if len(lang_code) == 2:
                    return lang_code, confidence
                else:
                    return 'en', 0.5
        
        except Exception as e:
            logger.warning(f"FastText confidence detection hatası: {e}, fallback kullanılıyor")
            return self._detect_with_heuristic_confidence(text)
    
    def _detect_with_heuristic_confidence(self, text: str) -> Tuple[str, float]:
        """
        Basit heuristik ile dil algılama ve güven skoru (fallback)
        
        Args:
            text: Algılanacak metin
        
        Returns:
            tuple: (dil, güven_skoru)
        """
        text_lower = text.lower()
        
        # Türkçe karakter sayısı
        turkish_char_count = sum(1 for char in text if char in self.TURKISH_CHARS)
        total_chars = len([c for c in text if c.isalpha()])
        turkish_char_ratio = turkish_char_count / (total_chars + 1e-9)
        
        # Türkçe kelime sayısı
        words = set(re.findall(r'\b\w+\b', text_lower))
        turkish_word_count = len(words.intersection(self.TURKISH_COMMON_WORDS))
        total_words = len(words)
        turkish_word_ratio = turkish_word_count / (total_words + 1e-9)
        
        # Güven skoru hesapla
        confidence = min(1.0, (turkish_char_ratio * 0.6 + turkish_word_ratio * 0.4) * 2)
        
        # Karar
        if turkish_char_count > 0 or turkish_word_count >= 2:
            return 'tr', confidence
        else:
            return 'en', 1.0 - confidence


# Backward compatibility: SimpleLanguageDetector alias
SimpleLanguageDetector = LanguageDetector

