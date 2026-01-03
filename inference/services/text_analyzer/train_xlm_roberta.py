"""
XLM-RoBERTa Fine-tuning Script for Disaster Relevance Classification
Multilingual model eğitimi (Türkçe + İngilizce)
"""
import sys
import codecs
import json
from pathlib import Path
from collections import Counter
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from datasets import Dataset as HFDataset

from data_processor import DataProcessor

# Windows için Unicode desteği
if sys.platform == 'win32':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')


class DisasterDataset(Dataset):
    """PyTorch Dataset for disaster classification"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def compute_metrics(eval_pred):
    """Metrics hesaplama fonksiyonu"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='binary', zero_division=0)
    recall = recall_score(labels, predictions, average='binary', zero_division=0)
    f1 = f1_score(labels, predictions, average='binary', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def train_xlm_roberta(
    model_name: str = "xlm-roberta-base",
    output_dir: str = None,
    max_length: int = 512,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    num_epochs: int = 3,
    use_gpu: bool = True
):
    """
    XLM-RoBERTa model eğitimi
    
    Args:
        model_name: HuggingFace model adı
        output_dir: Model kayıt dizini
        max_length: Maximum sequence length
        batch_size: Batch size
        learning_rate: Learning rate
        num_epochs: Epoch sayısı
        use_gpu: GPU kullanımı
    """
    print("="*70)
    print("XLM-ROBERTA DISASTER RELEVANCE CLASSIFICATION TRAINING")
    print("="*70)
    
    # GPU kontrolü
    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    print(f"\n[INFO] Device: {device}")
    
    # Output dizini
    if output_dir is None:
        models_dir = Path(__file__).parent / "models"
        output_dir = models_dir / "xlm_roberta"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # [1] Dataset yükleme
    print("\n[1/7] Datasetler yukleniyor...")
    processor = DataProcessor()
    texts, labels, sources = processor.process_all_datasets()
    
    if len(texts) == 0:
        print("[ERROR] Hiç veri yuklenemedi!")
        return
    
    print(f"  - Toplam kayit: {len(texts)}")
    print(f"  - Disaster related: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
    print(f"  - Not related: {len(labels) - sum(labels)} ({(len(labels) - sum(labels))/len(labels)*100:.1f}%)")
    
    # [2] Train-test split
    print(f"\n[2/7] Train-test split yapiliyor...")
    X_train, X_test, y_train, y_test, sources_train, sources_test = train_test_split(
        texts, labels, sources, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    
    print(f"  - Train set: {len(X_train)} kayit")
    print(f"  - Validation set: {len(X_val)} kayit")
    print(f"  - Test set: {len(X_test)} kayit")
    
    # [3] Tokenizer yükleme
    print(f"\n[3/7] Tokenizer yukleniyor: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # [4] Model yükleme
    print(f"\n[4/7] Model yukleniyor: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        problem_type="single_label_classification"
    )
    model.to(device)
    
    # [5] Dataset oluşturma
    print(f"\n[5/7] Datasets hazirlaniyor...")
    
    # HuggingFace Dataset formatına çevir
    train_dataset = HFDataset.from_dict({
        'text': X_train,
        'label': y_train
    })
    
    val_dataset = HFDataset.from_dict({
        'text': X_val,
        'label': y_val
    })
    
    test_dataset = HFDataset.from_dict({
        'text': X_test,
        'label': y_test
    })
    
    # Tokenization
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=max_length
        )
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    # Rename 'label' to 'labels' (Trainer expects 'labels')
    train_dataset = train_dataset.rename_column('label', 'labels')
    val_dataset = val_dataset.rename_column('label', 'labels')
    test_dataset = test_dataset.rename_column('label', 'labels')
    
    # Format for PyTorch
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    # [6] Training arguments
    print(f"\n[6/7] Training arguments ayarlaniyor...")
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=500,
        logging_dir=str(output_dir / "logs"),
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),  # Mixed precision training (GPU varsa)
        dataloader_num_workers=0 if sys.platform == 'win32' else 4,
    )
    
    # [7] Trainer oluşturma ve eğitim
    print(f"\n[7/7] Model egitimi basliyor...")
    print("="*70)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Eğitim
    train_result = trainer.train()
    
    # En iyi modeli kaydet
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    print("\n" + "="*70)
    print("EGITIM TAMAMLANDI!")
    print("="*70)
    print(f"\nTraining Loss: {train_result.training_loss:.4f}")
    
    # [8] Test set evaluation
    print("\n" + "="*70)
    print("TEST SET EVALUATION")
    print("="*70)
    
    test_predictions = trainer.predict(test_dataset)
    test_preds = np.argmax(test_predictions.predictions, axis=1)
    test_labels = test_predictions.label_ids
    
    test_accuracy = accuracy_score(test_labels, test_preds)
    test_precision = precision_score(test_labels, test_preds, average='binary', zero_division=0)
    test_recall = recall_score(test_labels, test_preds, average='binary', zero_division=0)
    test_f1 = f1_score(test_labels, test_preds, average='binary', zero_division=0)
    
    print(f"\nTest Set Metrics:")
    print(f"  - Accuracy:  {test_accuracy:.4f}")
    print(f"  - Precision: {test_precision:.4f}")
    print(f"  - Recall:    {test_recall:.4f}")
    print(f"  - F1-Score:  {test_f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(test_labels, test_preds)
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {cm[0][0]}")
    print(f"  False Positives: {cm[0][1]}")
    print(f"  False Negatives: {cm[1][0]}")
    print(f"  True Positives:  {cm[1][1]}")
    
    # [9] Metadata kaydetme
    print(f"\n[9/9] Metadata kaydediliyor...")
    metadata = {
        "model_name": model_name,
        "model_type": "XLM-RoBERTa",
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test),
        "total_samples": len(texts),
        "metrics": {
            "train": {
                "loss": float(train_result.training_loss)
            },
            "test": {
                "accuracy": float(test_accuracy),
                "precision": float(test_precision),
                "recall": float(test_recall),
                "f1": float(test_f1)
            }
        },
        "hyperparameters": {
            "max_length": max_length,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "device": str(device)
        },
        "data_metadata": {
            "disaster_related": sum(labels),
            "not_related": len(labels) - sum(labels),
            "label_distribution": {
                "disaster_related_pct": (sum(labels) / len(labels)) * 100,
                "not_related_pct": ((len(labels) - sum(labels)) / len(labels)) * 100
            }
        }
    }
    
    metadata_path = output_dir / "model_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"  [OK] Model kaydedildi: {output_dir}")
    print(f"  [OK] Metadata kaydedildi: {metadata_path}")
    
    print("\n" + "="*70)
    print("ISLEM TAMAMLANDI!")
    print("="*70)
    
    return model, tokenizer, metadata


if __name__ == "__main__":
    # Eğitim parametreleri
    MODEL_NAME = "xlm-roberta-base"  # veya "xlm-roberta-large" (daha büyük, daha iyi performans)
    
    # GPU kontrolü
    USE_GPU = torch.cuda.is_available()
    if USE_GPU:
        print(f"[INFO] GPU bulundu: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("[WARNING] GPU bulunamadi, CPU kullanilacak (cok yavas olabilir)")
    
    # Eğitimi başlat
    train_xlm_roberta(
        model_name=MODEL_NAME,
        max_length=512,
        batch_size=16 if USE_GPU else 4,  # GPU yoksa batch size küçült
        learning_rate=2e-5,
        num_epochs=3,
        use_gpu=USE_GPU
    )





