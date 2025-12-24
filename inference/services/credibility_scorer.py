"""
Credibility Scorer Service
T7: Final Credibility Scoring
Tüm sinyalleri birleştirir
"""
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class AnalysisSignals:
    """ML analiz sonuçları"""
    text_relevance: float
    help_request_probability: float
    image_damage_score: Optional[float] = None
    consistency_score: Optional[float] = None
    reuse_score: Optional[float] = None
    author_trust_score: float = 0.5


class CredibilityScorer:
    """
    Tüm ML sinyallerini birleştirip final credibility score hesaplar
    """
    
    def __init__(self):
        # Ağırlıklar - bu değerler validation set üzerinde optimize edilecek
        self.weights = {
            "text_relevance": 0.25,
            "help_request": 0.20,
            "image_damage": 0.15,
            "consistency": 0.15,
            "reuse_penalty": -0.20,  # Negatif - yüksek reuse score düşük credibility
            "author_trust": 0.20
        }
    
    def compute_full_score(self, signals: AnalysisSignals) -> Dict:
        """
        Final credibility score hesapla
        
        Args:
            signals: Tüm ML analiz sonuçları
            
        Returns:
            {
                "credibility_score": float,  # 0.0 - 1.0
                "breakdown": dict  # Detaylı skorlar
            }
        """
        score = 0.0
        
        # Text relevance
        score += signals.text_relevance * self.weights["text_relevance"]
        
        # Help request (daha yüksek = daha credible)
        score += signals.help_request_probability * self.weights["help_request"]
        
        # Image damage (varsa)
        if signals.image_damage_score is not None:
            score += signals.image_damage_score * self.weights["image_damage"]
        
        # Text-image consistency (varsa)
        if signals.consistency_score is not None:
            score += signals.consistency_score * self.weights["consistency"]
        
        # Image reuse penalty (yüksek reuse = düşük credibility)
        if signals.reuse_score is not None:
            score += (1.0 - signals.reuse_score) * abs(self.weights["reuse_penalty"])
        
        # Author trust
        score += signals.author_trust_score * self.weights["author_trust"]
        
        # Normalize to [0, 1]
        credibility_score = max(0.0, min(1.0, score))
        
        return {
            "credibility_score": credibility_score,
            "breakdown": {
                "text_relevance": signals.text_relevance,
                "help_request_probability": signals.help_request_probability,
                "image_damage_score": signals.image_damage_score,
                "consistency_score": signals.consistency_score,
                "reuse_score": signals.reuse_score,
                "author_trust_score": signals.author_trust_score
            }
        }









