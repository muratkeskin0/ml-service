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

from data_processor import DataProcessor


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
    X_train, X_test, y_train, y_test, sources_train, sources_test = train_test_split(
        texts, labels, sources, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"  - Train set: {len(X_train)} kayit")
    print(f"  - Test set: {len(X_test)} kayit")
    
    # TF-IDF Vectorization
    print(f"\n[4/5] TF-IDF vectorization yapiliyor...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),  # Unigrams ve bigrams
        min_df=2,  # En az 2 dokümanda geçen kelimeler
        max_df=0.95,  # En fazla %95 dokümanda geçen kelimeler
        stop_words='english'
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"  - Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"  - Feature matrix shape (train): {X_train_tfidf.shape}")
    print(f"  - Feature matrix shape (test): {X_test_tfidf.shape}")
    
    # Model eğitimi
    print(f"\n[5/5] Model egitiliyor...")
    model = LogisticRegression(
        C=1.0,
        class_weight='balanced',  # Imbalanced data için
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    )
    
    model.fit(X_train_tfidf, y_train)
    print("  [OK] Model egitimi tamamlandi")
    
    # Evaluation
    print(f"\n{'='*60}")
    print("MODEL EVALUATION")
    print(f"{'='*60}")
    
    # Train predictions
    y_train_pred = model.predict(X_train_tfidf)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    
    # Test predictions
    y_test_pred = model.predict(X_test_tfidf)
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
    
    # Model ve vectorizer'ı kaydet
    model_path = models_dir / "model.pkl"
    vectorizer_path = models_dir / "vectorizer.pkl"
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  [OK] Model kaydedildi: {model_path}")
    
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"  [OK] Vectorizer kaydedildi: {vectorizer_path}")
    
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
            "model_C": 1.0,
            "model_class_weight": "balanced"
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









