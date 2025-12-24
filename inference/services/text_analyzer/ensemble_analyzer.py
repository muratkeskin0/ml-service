"""
Ensemble Text Analyzer
Hem Logistic Regression hem de XLM-RoBERTa modellerini birlikte kullanarak
daha güvenilir tahminler yapar
"""
from typing import Tuple, Optional, Dict
from pathlib import Path
from text_analyzer import TextAnalyzer


class EnsembleTextAnalyzer:
    """
    Ensemble Text Analyzer
    İki modelin sonuçlarını birleştirerek daha robust tahminler yapar
    """
    
    def __init__(
        self, 
        model_path: Optional[str] = None,
        logistic_weight: float = 0.3,
        xlm_roberta_weight: float = 0.7
    ):
        """
        Ensemble analyzer başlat
        
        Args:
            model_path: Model klasör yolu (opsiyonel)
            logistic_weight: Logistic Regression modelinin ağırlığı (0.0-1.0)
            xlm_roberta_weight: XLM-RoBERTa modelinin ağırlığı (0.0-1.0)
                Not: Ağırlıklar normalize edilecek, toplamları 1.0 olması gerekmez
        """
        if model_path:
            model_dir = Path(model_path)
        else:
            # Varsayılan model yolu
            current_file = Path(__file__)
            model_dir = current_file.parent / "models"
        
        # Her iki modeli de yükle
        print("[INFO] Ensemble analyzer başlatılıyor...")
        
        try:
            self.logistic_analyzer = TextAnalyzer(
                model_path=str(model_dir),
                model_type='logistic'
            )
            print("[OK] Logistic Regression modeli yüklendi")
        except Exception as e:
            print(f"[WARNING] Logistic Regression yüklenemedi: {e}")
            self.logistic_analyzer = None
        
        try:
            self.xlm_roberta_analyzer = TextAnalyzer(
                model_path=str(model_dir),
                model_type='xlm_roberta'
            )
            print("[OK] XLM-RoBERTa modeli yüklendi")
        except Exception as e:
            print(f"[WARNING] XLM-RoBERTa yüklenemedi: {e}")
            self.xlm_roberta_analyzer = None
        
        # En az bir model yüklenmiş olmalı
        if not self.logistic_analyzer and not self.xlm_roberta_analyzer:
            raise RuntimeError("Hiçbir model yüklenemedi! En az bir model gerekli.")
        
        # Ağırlıkları normalize et
        total_weight = logistic_weight + xlm_roberta_weight
        if total_weight > 0:
            self.logistic_weight = logistic_weight / total_weight
            self.xlm_roberta_weight = xlm_roberta_weight / total_weight
        else:
            # Eşit ağırlık
            self.logistic_weight = 0.5
            self.xlm_roberta_weight = 0.5
        
        # Mevcut modellere göre ağırlıkları ayarla
        if not self.logistic_analyzer:
            self.logistic_weight = 0.0
            self.xlm_roberta_weight = 1.0
        elif not self.xlm_roberta_analyzer:
            self.logistic_weight = 1.0
            self.xlm_roberta_weight = 0.0
        
        print(f"[INFO] Ensemble ağırlıkları - Logistic: {self.logistic_weight:.2f}, XLM-RoBERTa: {self.xlm_roberta_weight:.2f}")
    
    def classify_disaster_relevance(self, text: str) -> Tuple[bool, float, Dict]:
        """
        Ensemble ile disaster relevance classification
        
        Args:
            text: Analiz edilecek metin
            
        Returns:
            Tuple[bool, float, Dict]: 
                - is_disaster_related: Final tahmin (True/False)
                - relevance_score: Final güven skoru (0.0-1.0)
                - details: Detaylı sonuçlar (her modelin sonuçları)
        """
        if not text or len(text.strip()) == 0:
            return False, 0.0, {}
        
        results = {}
        scores = []
        predictions = []
        
        # Logistic Regression sonucu
        if self.logistic_analyzer:
            try:
                is_related, score = self.logistic_analyzer.classify_disaster_relevance(text)
                results['logistic'] = {
                    'is_disaster_related': is_related,
                    'relevance_score': score,
                    'weight': self.logistic_weight
                }
                scores.append(score * self.logistic_weight)
                predictions.append((1 if is_related else 0) * self.logistic_weight)
            except Exception as e:
                print(f"[WARNING] Logistic Regression tahmini başarısız: {e}")
                results['logistic'] = {'error': str(e)}
        
        # XLM-RoBERTa sonucu
        if self.xlm_roberta_analyzer:
            try:
                is_related, score = self.xlm_roberta_analyzer.classify_disaster_relevance(text)
                results['xlm_roberta'] = {
                    'is_disaster_related': is_related,
                    'relevance_score': score,
                    'weight': self.xlm_roberta_weight
                }
                scores.append(score * self.xlm_roberta_weight)
                predictions.append((1 if is_related else 0) * self.xlm_roberta_weight)
            except Exception as e:
                print(f"[WARNING] XLM-RoBERTa tahmini başarısız: {e}")
                results['xlm_roberta'] = {'error': str(e)}
        
        # Ensemble sonuçları hesapla
        if not scores:
            return False, 0.0, results
        
        # Weighted average score
        final_score = sum(scores)
        
        # Weighted voting (prediction)
        final_prediction_sum = sum(predictions)
        threshold = (self.logistic_weight + self.xlm_roberta_weight) / 2.0
        is_related = final_prediction_sum >= threshold
        
        # Detayları ekle
        results['ensemble'] = {
            'final_is_disaster_related': is_related,
            'final_relevance_score': final_score,
            'method': 'weighted_average'
        }
        
        return is_related, final_score, results
    
    def classify_disaster_relevance_simple(self, text: str) -> Tuple[bool, float]:
        """
        Basit ensemble classification (sadece final sonuç)
        
        Args:
            text: Analiz edilecek metin
            
        Returns:
            Tuple[bool, float]: (is_disaster_related, relevance_score)
        """
        is_related, score, _ = self.classify_disaster_relevance(text)
        return is_related, score


