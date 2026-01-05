"""
Free Translation Service (HuggingFace MarianMT)
Tamamen ücretsiz, offline çalışan çeviri servisi
Türkçe → İngilizce çeviri için Helsinki-NLP/opus-mt-tr-en modeli kullanır
Post-processing ile çeviri kalitesi iyileştirilir
Not: Diğer diller için model değiştirilebilir (es, fr, de, ar, ru, vb.)
"""
import logging
from typing import Optional
from functools import lru_cache

logger = logging.getLogger(__name__)

# Post-processor import
try:
    from .translation_postprocessor import TranslationPostProcessor
    POSTPROCESSOR_AVAILABLE = True
except ImportError:
    try:
        from translation_postprocessor import TranslationPostProcessor
        POSTPROCESSOR_AVAILABLE = True
    except ImportError:
        POSTPROCESSOR_AVAILABLE = False
        logger.warning("Translation post-processor bulunamadı, post-processing devre dışı")

# HuggingFace Transformers import
try:
    from transformers import MarianMTModel, MarianTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers modülü bulunamadı. Çeviri özelliği kullanılamayacak.")
    logger.warning("Yüklemek için: pip install transformers torch")


class FreeTranslationService:
    """
    Ücretsiz çeviri servisi (HuggingFace MarianMT)
    Tamamen offline çalışır, API key gerekmez
    Türkçe → İngilizce çeviri için optimize edilmiş
    """
    
    def __init__(self, use_cache: bool = True, max_cache_size: int = 1000, device: Optional[str] = None):
        """
        Free translation service başlat
        
        Args:
            use_cache: Cache kullanılsın mı?
            max_cache_size: Maksimum cache boyutu
            device: 'cuda' (GPU) veya 'cpu' (varsayılan)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers modülü bulunamadı. "
                "Yüklemek için: pip install transformers torch"
            )
        
        # Device belirleme
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Model ve tokenizer yükleme
        # Türkçe-İngilizce özel model (daha iyi kalite)
        # Alternatifler:
        # - "Helsinki-NLP/opus-mt-tr-en" (önerilen, küçük ve hızlı)
        # - "Helsinki-NLP/opus-mt-tc-big-tr-en" (daha büyük, daha iyi kalite)
        # - "Helsinki-NLP/opus-mt-mul-en" (multi-language, düşük kalite - kullanmayın)
        self.model_name = "Helsinki-NLP/opus-mt-tr-en"  # Türkçe → İngilizce (özel model)
        self.model = None
        self.tokenizer = None
        self._load_model()
        
        # Cache
        self.use_cache = use_cache
        self.cache = {} if use_cache else None
        self.max_cache_size = max_cache_size
        self.translation_count = 0
        self.cache_hits = 0
        
        # Post-processor
        self.postprocessor = None
        if POSTPROCESSOR_AVAILABLE:
            try:
                self.postprocessor = TranslationPostProcessor()
                logger.info("Translation post-processor aktif")
            except Exception as e:
                logger.warning(f"Post-processor yüklenemedi: {e}")
        else:
            logger.info("Translation post-processor devre dışı")
        
        logger.info(f"Free translation service yüklendi (device: {self.device})")
    
    def _load_model(self):
        """MarianMT modelini yükle"""
        try:
            logger.info(f"MarianMT Türkçe-İngilizce modeli yükleniyor: {self.model_name}")
            logger.info("İlk kullanımda model indirilecek (~500 MB)...")
            logger.info("Türkçe → İngilizce çeviri için optimize edilmiş")
            
            self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
            self.model = MarianMTModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()  # Evaluation mode
            
            logger.info(f"Model yüklendi ve {self.device} üzerinde çalışıyor")
        except Exception as e:
            logger.error(f"Model yüklenirken hata: {e}")
            raise
    
    def translate_to_english(self, text: str, source_lang: str = 'auto') -> str:
        """
        Metni İngilizce'ye çevir (Türkçe → İngilizce)
        
        Args:
            text: Çevrilecek metin
            source_lang: Kaynak dil ISO 639-1 kodu
                - 'tr': Türkçe (çeviri yapılır)
                - 'en': İngilizce (çeviri yapılmaz)
                - 'auto': Otomatik algılama (language detector ile)
        
        Returns:
            str: Çevrilmiş metin (İngilizce)
                Hata durumunda orijinal metin döner
        
        Not: 
            - Model: Helsinki-NLP/opus-mt-tr-en (Türkçe-İngilizce özel model)
            - Sadece Türkçe metinler için optimize edilmiş
            - Diğer diller için farklı model gerekir
        """
        if not text or len(text.strip()) == 0:
            return text
        
        # Zaten İngilizce ise çevirme
        if source_lang == 'en' or source_lang.lower() == 'english':
            return text
        
        # Cache kontrolü
        if self.use_cache and text in self.cache:
            self.cache_hits += 1
            return self.cache[text]
        
        try:
            # Tokenization
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Translation
            with torch.no_grad():  # Memory efficient
                translated = self.model.generate(**inputs)
            
            # Decode
            translated_text = self.tokenizer.decode(translated[0], skip_special_tokens=True)
            
            # Post-processing (çeviri kalitesini iyileştir)
            if self.postprocessor:
                try:
                    original_translated = translated_text
                    translated_text = self.postprocessor.improve_translation(translated_text, text)
                    if self.postprocessor.is_improved(original_translated, translated_text):
                        logger.debug(f"Çeviri iyileştirildi: '{original_translated[:50]}...' -> '{translated_text[:50]}...'")
                except Exception as e:
                    logger.warning(f"Post-processing hatası: {e}, orijinal çeviri kullanılıyor")
            
            # Cache'e ekle
            if self.use_cache:
                if len(self.cache) >= self.max_cache_size:
                    # En eski entry'yi sil (basit FIFO)
                    first_key = next(iter(self.cache))
                    del self.cache[first_key]
                
                self.cache[text] = translated_text
            
            self.translation_count += 1
            
            logger.debug(f"Çeviri yapıldı: '{text[:50]}...' -> '{translated_text[:50]}...'")
            
            return translated_text
        
        except Exception as e:
            logger.error(f"Çeviri hatası: {e}, Orijinal metin kullanılıyor")
            # Fallback: Orijinal metni döndür
            return text
    
    def get_stats(self) -> dict:
        """
        Çeviri istatistiklerini al
        
        Returns:
            dict: İstatistikler
        """
        stats = {
            'translation_count': self.translation_count,
            'cache_hits': self.cache_hits,
            'cache_size': len(self.cache) if self.cache else 0,
            'cache_hit_rate': (
                self.cache_hits / (self.translation_count + self.cache_hits)
                if (self.translation_count + self.cache_hits) > 0
                else 0.0
            ),
            'device': self.device,
            'model': self.model_name
        }
        return stats
    
    def clear_cache(self):
        """Cache'i temizle"""
        if self.cache:
            self.cache.clear()
            logger.info("Çeviri cache'i temizlendi")


# Backward compatibility: TranslationService alias
TranslationService = FreeTranslationService

