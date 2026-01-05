"""
Feature Extractor for Disaster Relevance Classification
Linguistic, temporal, and other features extraction
"""
import re
import numpy as np
from typing import Dict, List
from datetime import datetime


class FeatureExtractor:
    """
    Feature extraction for disaster relevance classification
    Extracts linguistic, temporal, and other features from text
    """
    
    def __init__(self):
        """Initialize feature extractor"""
        pass
    
    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """
        Extract linguistic features from text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of linguistic features
        """
        if not text:
            return self._empty_features()
        
        features = {}
        
        # Basic text statistics
        features['char_count'] = float(len(text))
        features['word_count'] = float(len(text.split()))
        features['sentence_count'] = float(len(re.split(r'[.!?]+', text)))
        
        # Hashtag and mention counts
        features['hashtag_count'] = float(text.count('#'))
        features['mention_count'] = float(text.count('@'))
        
        # URL count
        url_pattern = r'http\S+|www\.\S+'
        urls = re.findall(url_pattern, text)
        features['url_count'] = float(len(urls))
        
        # Punctuation counts
        features['exclamation_count'] = float(text.count('!'))
        features['question_count'] = float(text.count('?'))
        features['period_count'] = float(text.count('.'))
        features['comma_count'] = float(text.count(','))
        
        # Capitalization
        if len(text) > 0:
            uppercase_chars = sum(1 for c in text if c.isupper())
            features['uppercase_ratio'] = float(uppercase_chars) / len(text)
            features['uppercase_count'] = float(uppercase_chars)
        else:
            features['uppercase_ratio'] = 0.0
            features['uppercase_count'] = 0.0
        
        # Digit count
        features['digit_count'] = float(sum(1 for c in text if c.isdigit()))
        
        # Special characters
        features['special_char_count'] = float(len(re.findall(r'[^a-zA-Z0-9\s]', text)))
        
        # Average word length
        words = text.split()
        if words:
            features['avg_word_length'] = float(np.mean([len(word) for word in words]))
        else:
            features['avg_word_length'] = 0.0
        
        # Disaster-related keywords (basic)
        disaster_keywords = [
            'disaster', 'emergency', 'crisis', 'earthquake', 'flood', 'fire',
            'hurricane', 'tornado', 'tsunami', 'volcano', 'landslide',
            'rescue', 'evacuate', 'damage', 'casualty', 'victim',
            'afet', 'deprem', 'sel', 'yangın', 'acil', 'kurtarma'
        ]
        text_lower = text.lower()
        keyword_count = sum(1 for keyword in disaster_keywords if keyword in text_lower)
        features['disaster_keyword_count'] = float(keyword_count)
        features['disaster_keyword_ratio'] = float(keyword_count) / len(words) if words else 0.0
        
        return features
    
    def extract_temporal_features(self, timestamp: str = None) -> Dict[str, float]:
        """
        Extract temporal features from timestamp
        
        Args:
            timestamp: ISO format timestamp string (optional)
            
        Returns:
            Dictionary of temporal features
        """
        features = {}
        
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                features['hour'] = float(dt.hour)
                features['day_of_week'] = float(dt.weekday())  # 0=Monday, 6=Sunday
                features['day_of_month'] = float(dt.day)
                features['month'] = float(dt.month)
                features['is_weekend'] = float(1.0 if dt.weekday() >= 5 else 0.0)
                features['is_night'] = float(1.0 if dt.hour >= 22 or dt.hour < 6 else 0.0)
                features['is_business_hours'] = float(1.0 if 9 <= dt.hour < 17 else 0.0)
            except:
                # If timestamp parsing fails, return zeros
                features = self._empty_temporal_features()
        else:
            features = self._empty_temporal_features()
        
        return features
    
    def extract_all_features(self, text: str, timestamp: str = None) -> Dict[str, float]:
        """
        Extract all features (linguistic + temporal)
        
        Args:
            text: Input text
            timestamp: ISO format timestamp string (optional)
            
        Returns:
            Dictionary of all features
        """
        features = {}
        
        # Linguistic features
        linguistic = self.extract_linguistic_features(text)
        features.update(linguistic)
        
        # Temporal features
        temporal = self.extract_temporal_features(timestamp)
        features.update(temporal)
        
        return features
    
    def _empty_features(self) -> Dict[str, float]:
        """Return empty linguistic features"""
        return {
            'char_count': 0.0,
            'word_count': 0.0,
            'sentence_count': 0.0,
            'hashtag_count': 0.0,
            'mention_count': 0.0,
            'url_count': 0.0,
            'exclamation_count': 0.0,
            'question_count': 0.0,
            'period_count': 0.0,
            'comma_count': 0.0,
            'uppercase_ratio': 0.0,
            'uppercase_count': 0.0,
            'digit_count': 0.0,
            'special_char_count': 0.0,
            'avg_word_length': 0.0,
            'disaster_keyword_count': 0.0,
            'disaster_keyword_ratio': 0.0
        }
    
    def _empty_temporal_features(self) -> Dict[str, float]:
        """Return empty temporal features"""
        return {
            'hour': 0.0,
            'day_of_week': 0.0,
            'day_of_month': 0.0,
            'month': 0.0,
            'is_weekend': 0.0,
            'is_night': 0.0,
            'is_business_hours': 0.0
        }
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of all feature names
        
        Returns:
            List of feature names
        """
        linguistic_names = list(self._empty_features().keys())
        temporal_names = list(self._empty_temporal_features().keys())
        return linguistic_names + temporal_names



