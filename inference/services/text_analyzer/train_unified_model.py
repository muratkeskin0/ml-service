"""
Unified Model Training Script
Tüm datasetleri kullanarak disaster relevance classification modeli eğitir
"""
import sys
import codecs
import json
import pickle
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy.sparse import hstack

from data_processor import DataProcessor
try:
    from feature_extractor import FeatureExtractor
except ImportError:
    print("[ERROR] feature_extractor modülü bulunamadı!")
    FeatureExtractor = None


def train_model():
    """Model eğitimi"""
    print("="*60)
    print("DISASTER RELEVANCE CLASSIFICATION MODEL TRAINING")
    print("="*60)
    
    # Windows için Unicode desteği
    if sys.platform == 'win32':
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    
    # Data processor'ı başlat
    processor = DataProcessor()
    
    # Tüm datasetleri yükle
    print("\n[1/5] Datasetler yukleniyor...")
    texts, labels, sources = processor.process_all_datasets()
    
    if len(texts) == 0:
        print("[ERROR] Hiç veri yuklenemedi!")
        return
    
    print(f"\n[2/5] Veri hazirlaniyor...")
    print(f"  - Toplam kayit: {len(texts)}")
    print(f"  - Disaster related: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
    print(f"  - Not related: {len(labels) - sum(labels)} ({(len(labels) - sum(labels))/len(labels)*100:.1f}%)")
    
    # Source dağılımını hesapla
    source_counts = Counter(sources)
    print(f"\n  Dataset kaynaklari:")
    for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    - {source}: {count}")
    
    # Train-test split
    print(f"\n[3/5] Train-test split yapiliyor...")
    # Veri kalitesi kontrolü: Çok kısa metinleri filtrele
    min_length = 10  # Minimum karakter sayısı
    filtered_texts = []
    filtered_labels = []
    filtered_sources = []
    
    for text, label, source in zip(texts, labels, sources):
        if len(text.strip()) >= min_length:
            filtered_texts.append(text)
            filtered_labels.append(label)
            filtered_sources.append(source)
    
    print(f"  - Filtreleme öncesi: {len(texts)} kayit")
    print(f"  - Filtreleme sonrası: {len(filtered_texts)} kayit (min {min_length} karakter)")
    
    X_train, X_test, y_train, y_test, sources_train, sources_test = train_test_split(
        filtered_texts, filtered_labels, filtered_sources, 
        test_size=0.2, random_state=42, stratify=filtered_labels
    )
    
    print(f"  - Train set: {len(X_train)} kayit")
    print(f"  - Test set: {len(X_test)} kayit")
    
    # Feature Extraction
    print(f"\n[4/6] Feature extraction yapiliyor...")
    
    # TF-IDF Vectorization
    print(f"  [4.1/6] TF-IDF vectorization yapiliyor...")
    # Word-level n-grams (unigrams + bigrams)
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),  # Unigrams ve bigrams
        min_df=2,  # En az 2 dokümanda geçen kelimeler
        max_df=0.95,  # En fazla %95 dokümanda geçen kelimeler
        stop_words='english'
    )
    
    # Character-level n-grams (3-4 karakterlik n-grams) - Typo tolerance için
    char_vectorizer = TfidfVectorizer(
        max_features=5000,  # Character n-grams için daha az feature
        analyzer='char_wb',  # Character n-grams (word boundaries)
        ngram_range=(3, 4),  # 3-4 karakterlik n-grams
        min_df=2,
        max_df=0.95
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"    - Word-level vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"    - Word-level TF-IDF matrix shape (train): {X_train_tfidf.shape}")
    print(f"    - Word-level TF-IDF matrix shape (test): {X_test_tfidf.shape}")
    
    # Character-level n-grams
    print(f"    - Character-level n-grams extraction...")
    X_train_char = char_vectorizer.fit_transform(X_train)
    X_test_char = char_vectorizer.transform(X_test)
    
    print(f"    - Character-level vocabulary size: {len(char_vectorizer.vocabulary_)}")
    print(f"    - Character-level TF-IDF matrix shape (train): {X_train_char.shape}")
    print(f"    - Character-level TF-IDF matrix shape (test): {X_test_char.shape}")
    
    # Combine word-level and character-level TF-IDF
    X_train_tfidf = hstack([X_train_tfidf, X_train_char])
    X_test_tfidf = hstack([X_test_tfidf, X_test_char])
    
    print(f"    - Combined TF-IDF matrix shape (train): {X_train_tfidf.shape}")
    print(f"    - Combined TF-IDF matrix shape (test): {X_test_tfidf.shape}")
    
    # Linguistic Features
    print(f"  [4.2/6] Linguistic features extraction yapiliyor...")
    if FeatureExtractor is None:
        print("[WARNING] FeatureExtractor bulunamadı, linguistic features atlanıyor")
        train_linguistic_array = np.zeros((len(X_train), 17))
        test_linguistic_array = np.zeros((len(X_test), 17))
    else:
        feature_extractor = FeatureExtractor()
        
        # Extract linguistic features for train set
        train_linguistic_features = []
        for text in X_train:
            features = feature_extractor.extract_linguistic_features(text)
            train_linguistic_features.append([features[key] for key in feature_extractor.get_feature_names()[:17]])  # Only linguistic (17 features)
        
        train_linguistic_array = np.array(train_linguistic_features)
        
        # Extract linguistic features for test set
        test_linguistic_features = []
        for text in X_test:
            features = feature_extractor.extract_linguistic_features(text)
            test_linguistic_features.append([features[key] for key in feature_extractor.get_feature_names()[:17]])  # Only linguistic (17 features)
        
        test_linguistic_array = np.array(test_linguistic_features)
    
    print(f"    - Linguistic features count: {train_linguistic_array.shape[1]}")
    print(f"    - Linguistic features shape (train): {train_linguistic_array.shape}")
    print(f"    - Linguistic features shape (test): {test_linguistic_array.shape}")
    
    # Combine TF-IDF and linguistic features
    print(f"  [4.3/6] Features birlestiriliyor...")
    from scipy.sparse import csr_matrix
    
    train_linguistic_sparse = csr_matrix(train_linguistic_array)
    test_linguistic_sparse = csr_matrix(test_linguistic_array)
    
    X_train_combined = hstack([X_train_tfidf, train_linguistic_sparse])
    X_test_combined = hstack([X_test_tfidf, test_linguistic_sparse])
    
    print(f"    - Combined features shape (train): {X_train_combined.shape}")
    print(f"    - Combined features shape (test): {X_test_combined.shape}")
    print(f"    - Total features: {X_train_combined.shape[1]}")
    
    # Model eğitimi
    print(f"\n[5/6] Model egitiliyor...")
    model = LogisticRegression(
        C=1.0,
        class_weight='balanced',  # Imbalanced data için
        max_iter=3000,  # Artırıldı (convergence için)
        random_state=42,
        solver='lbfgs',
        verbose=1  # Progress göster
    )
    
    model.fit(X_train_combined, y_train)
    print("  [OK] Model egitimi tamamlandi")
    
    # Evaluation
    print(f"\n[6/6] Model evaluation yapiliyor...")
    print(f"{'='*60}")
    print("MODEL EVALUATION")
    print(f"{'='*60}")
    
    # Train predictions
    y_train_pred = model.predict(X_train_combined)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    
    # Test predictions
    y_test_pred = model.predict(X_test_combined)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    print(f"\nTrain Set Metrics:")
    print(f"  - Accuracy:  {train_accuracy:.4f}")
    print(f"  - Precision: {train_precision:.4f}")
    print(f"  - Recall:    {train_recall:.4f}")
    print(f"  - F1-Score:  {train_f1:.4f}")
    
    print(f"\nTest Set Metrics:")
    print(f"  - Accuracy:  {test_accuracy:.4f}")
    print(f"  - Precision: {test_precision:.4f}")
    print(f"  - Recall:    {test_recall:.4f}")
    print(f"  - F1-Score:  {test_f1:.4f}")
    
    # Confusion Matrix
    print(f"\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"  True Negatives:  {cm[0][0]}")
    print(f"  False Positives: {cm[0][1]}")
    print(f"  False Negatives: {cm[1][0]}")
    print(f"  True Positives:  {cm[1][1]}")
    
    # Model kaydetme
    print(f"\n{'='*60}")
    print("MODEL KAYDEDILIYOR...")
    print(f"{'='*60}")
    
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Model, vectorizer ve feature extractor'ı kaydet
    model_path = models_dir / "model.pkl"
    vectorizer_path = models_dir / "vectorizer.pkl"
    char_vectorizer_path = models_dir / "char_vectorizer.pkl"
    feature_extractor_path = models_dir / "feature_extractor.pkl"
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  [OK] Model kaydedildi: {model_path}")
    
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"  [OK] Word-level vectorizer kaydedildi: {vectorizer_path}")
    
    with open(char_vectorizer_path, 'wb') as f:
        pickle.dump(char_vectorizer, f)
    print(f"  [OK] Character-level vectorizer kaydedildi: {char_vectorizer_path}")
    
    if FeatureExtractor is not None and 'feature_extractor' in locals():
        with open(feature_extractor_path, 'wb') as f:
            pickle.dump(feature_extractor, f)
        print(f"  [OK] Feature extractor kaydedildi: {feature_extractor_path}")
    else:
        print(f"  [SKIP] Feature extractor kaydedilmedi (kullanılmıyor)")
    
    # Metadata kaydet
    metadata = {
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "data_sources": dict(source_counts),
        "metrics": {
            "train": {
                "accuracy": float(train_accuracy),
                "precision": float(train_precision),
                "recall": float(train_recall),
                "f1": float(train_f1)
            },
            "test": {
                "accuracy": float(test_accuracy),
                "precision": float(test_precision),
                "recall": float(test_recall),
                "f1": float(test_f1)
            }
        },
        "data_metadata": {
            "total_samples": len(texts),
            "disaster_related": sum(labels),
            "not_related": len(labels) - sum(labels),
            "sources": dict(source_counts),
            "label_distribution": {
                "disaster_related_pct": (sum(labels) / len(labels)) * 100,
                "not_related_pct": ((len(labels) - sum(labels)) / len(labels)) * 100
            }
        },
        "model_params": {
            "vectorizer_max_features": 10000,
            "vectorizer_ngram_range": [1, 2],
            "char_vectorizer_max_features": 5000,
            "char_vectorizer_ngram_range": [3, 4],
            "model_C": 1.0,
            "model_class_weight": "balanced",
            "linguistic_features_enabled": True,
            "linguistic_features_count": 17,
            "character_ngrams_enabled": True
        }
    }
    
    metadata_path = models_dir / "model_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Metadata kaydedildi: {metadata_path}")
    
    print(f"\n{'='*60}")
    print("EGITIM TAMAMLANDI!")
    print(f"{'='*60}")


if __name__ == "__main__":
    train_model()









