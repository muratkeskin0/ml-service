"""
Translation Post-Processor
Çeviri kalitesini iyileştirmek için post-processing yapar
"""
import re
from typing import Dict, List

class TranslationPostProcessor:
    """
    Çeviri sonrası iyileştirme sınıfı
    Disaster terimlerini ve yaygın hataları düzeltir
    """
    
    # Disaster terimleri sözlüğü (Türkçe → İngilizce)
    DISASTER_TERMS = {
        # Deprem
        "depression": "earthquake",
        "depressed": "earthquake",
        "depress": "earthquake",
        "earthquake occurred": "earthquake occurred",
        "there's been an earthquake": "earthquake occurred",
        "there was an earthquake": "earthquake occurred",
        
        # Yangın
        "you're out": "fire broke out",
        "there's a fire": "fire broke out",
        "fire broke out": "fire broke out",
        "on fire": "burning",
        "the house is on fire": "houses are burning",
        "the woods are on fire": "forests are burning",
        
        # Sel
        "saltwaters": "flood waters",
        "salt water": "flood water",
        "floodwater": "flood waters",
        "floodwaters": "flood waters",
        "floods the city": "flooded the city",
        "raids the city": "flooded the city",
        
        # Acil durum
        "i'm in a hurry": "emergency",
        "emergency": "emergency",
        "urgent": "urgent",
        "immediate": "urgent",
        
        # Yardım
        "immediate assistance": "urgent help",
        "immediate help": "urgent help",
        "we need immediate assistance": "urgent help needed",
        "we need immediate help": "urgent help needed",
        
        # Enkaz
        "debris": "debris",
        "under the debris": "trapped under debris",
        "under debris": "trapped under debris",
        
        # Yaralılar
        "wounded": "injured people",
        "injuries": "injured people",
        "there were injuries": "there are injured people",
        
        # Tahliye
        "evicted": "evacuated",
        "being evicted": "being evacuated",
        "evacuated": "evacuated",
        
        # Arama kurtarma
        "forward calls": "search and rescue",
        "search and rescue": "search and rescue",
        "survivors": "people trapped",
    }
    
    # Yaygın çeviri hataları
    COMMON_FIXES = {
        # Çoğul düzeltmeleri
        r"\bhouse\b": "houses",  # "house" → "houses" (çoğul bağlamda)
        r"\bwood\b": "forests",  # "woods" → "forests"
        
        # Fiil düzeltmeleri
        r"\bis on fire\b": "are burning",
        r"\bwas destroyed\b": "collapsed",
        r"\bwere destroyed\b": "collapsed",
        
        # Bağlaç düzeltmeleri
        r"\bpost-earthquake\b": "after the earthquake",
        r"\bafter the earthquake\b": "after the earthquake",
        
        # Noktalama düzeltmeleri
        r"\.\s*$": "",  # Son noktayı kaldır (bazı durumlarda)
    }
    
    def __init__(self):
        """Post-processor başlat"""
        pass
    
    def fix_disaster_terms(self, text: str) -> str:
        """
        Disaster terimlerini düzelt
        
        Args:
            text: Çevrilmiş metin
            
        Returns:
            str: Düzeltilmiş metin
        """
        text_lower = text.lower()
        fixed_text = text
        
        # Disaster terimlerini düzelt
        for wrong_term, correct_term in self.DISASTER_TERMS.items():
            # Case-insensitive replacement
            pattern = re.compile(re.escape(wrong_term), re.IGNORECASE)
            if pattern.search(text_lower):
                fixed_text = pattern.sub(correct_term, fixed_text, count=1)
                # İlk eşleşmeden sonra dur (daha fazla değişiklik yapma)
                break
        
        return fixed_text
    
    def apply_common_fixes(self, text: str) -> str:
        """
        Yaygın çeviri hatalarını düzelt
        
        Args:
            text: Çevrilmiş metin
            
        Returns:
            str: Düzeltilmiş metin
        """
        fixed_text = text
        
        # Yaygın hataları düzelt
        for pattern, replacement in self.COMMON_FIXES.items():
            fixed_text = re.sub(pattern, replacement, fixed_text, flags=re.IGNORECASE)
        
        return fixed_text
    
    def improve_translation(self, translated_text: str, original_turkish: str = None) -> str:
        """
        Çeviriyi iyileştir
        
        Args:
            translated_text: Çevrilmiş metin
            original_turkish: Orijinal Türkçe metin (opsiyonel, gelecekte kullanılabilir)
            
        Returns:
            str: İyileştirilmiş çeviri
        """
        # 1. Disaster terimlerini düzelt
        improved = self.fix_disaster_terms(translated_text)
        
        # 2. Yaygın hataları düzelt
        improved = self.apply_common_fixes(improved)
        
        # 3. Fazla boşlukları temizle
        improved = re.sub(r'\s+', ' ', improved).strip()
        
        return improved
    
    def is_improved(self, original: str, improved: str) -> bool:
        """
        Çevirinin iyileştirilip iyileştirilmediğini kontrol et
        
        Args:
            original: Orijinal çeviri
            improved: İyileştirilmiş çeviri
            
        Returns:
            bool: İyileştirme yapıldı mı?
        """
        return original.lower() != improved.lower()


