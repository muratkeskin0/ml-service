"""
RoBERTa Fine-Tuning for Disaster Relevance Classification
Aynı veriyi kullanır (DataProcessor) - Logistic Regression ile aynı train/test split.
Model models/roberta/ altına kaydedilir; inference'da use_roberta=True ile kullanılır.

Calistirma (ml-service/inference/services/text_analyzer/ dizininden):
  python train_roberta.py
  python train_roberta.py --epochs 4 --batch_size 8 --lr 3e-5
"""
import sys
import os
import codecs
import json
from pathlib import Path
from collections import Counter

# Script text_analyzer dizininde calismali (data_processor ve data/ icin)
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) != os.getcwd():
    os.chdir(_SCRIPT_DIR)
    sys.path.insert(0, str(_SCRIPT_DIR))

# Windows için Unicode
if sys.platform == "win32":
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EvalPrediction,
)

from data_processor import DataProcessor


# Sabitler
MAX_LENGTH = 512
MODEL_NAME = "roberta-base"
NUM_LABELS = 2  # 0: not_related, 1: disaster_related
ROBERTA_SUBDIR = "roberta"
MIN_TEXT_LENGTH = 10


class DisasterDataset(Dataset):
    """PyTorch Dataset for disaster relevance (text, label)."""

    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def compute_metrics(eval_pred: EvalPrediction):
    """F1, accuracy, precision, recall."""
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    preds = logits.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }


def train_roberta(
    output_dir: str = None,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    max_length: int = MAX_LENGTH,
):
    """
    RoBERTa-base'i disaster relevance için fine-tune et.
    Veri: DataProcessor.process_all_datasets() (Logistic Regression ile aynı).
    """
    print("=" * 60)
    print("ROBERTA FINE-TUNING - DISASTER RELEVANCE CLASSIFICATION")
    print("=" * 60)

    script_dir = Path(__file__).parent
    if output_dir is None:
        output_dir = script_dir / "models" / ROBERTA_SUBDIR
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Veri yükle (Logistic Regression ile aynı)
    print("\n[1/6] Datasetler yukleniyor (DataProcessor)...")
    processor = DataProcessor()
    texts, labels, sources = processor.process_all_datasets()
    if len(texts) == 0:
        raise RuntimeError("Hic veri yuklenemedi!")

    # 2) Filtreleme ve train/val split (train_unified_model ile uyumlu)
    print("\n[2/6] Filtreleme ve train/validation split...")
    filtered_texts, filtered_labels, filtered_sources = [], [], []
    for t, l, s in zip(texts, labels, sources):
        if len(t.strip()) >= MIN_TEXT_LENGTH:
            filtered_texts.append(t)
            filtered_labels.append(l)
            filtered_sources.append(s)

    X_train, X_val, y_train, y_val, _, _ = train_test_split(
        filtered_texts,
        filtered_labels,
        filtered_sources,
        test_size=0.15,
        random_state=42,
        stratify=filtered_labels,
    )
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}")
    print(f"  Disaster related (train): {sum(y_train)} ({100*sum(y_train)/len(y_train):.1f}%)")

    # 3) Tokenizer ve model
    print("\n[3/6] Tokenizer ve model yukleniyor...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
    )

    train_dataset = DisasterDataset(X_train, y_train, tokenizer, max_length)
    val_dataset = DisasterDataset(X_val, y_val, tokenizer, max_length)

    # 4) Training arguments
    print("\n[4/6] Training arguments ayarlaniyor...")
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=0.01,
        logging_dir=str(output_dir / "logs"),
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 5) Eğitim
    print("\n[5/6] Egitim basliyor...")
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # 6) Son değerlendirme ve metadata
    print("\n[6/6] Degerlendirme ve metadata kaydediliyor...")
    eval_result = trainer.evaluate()
    preds = trainer.predict(val_dataset)
    cm = confusion_matrix(y_val, preds.predictions.argmax(axis=1))

    metadata = {
        "model_name": MODEL_NAME,
        "num_labels": NUM_LABELS,
        "max_length": max_length,
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "metrics": {
            "validation": {
                "accuracy": eval_result.get("eval_accuracy", 0),
                "precision": eval_result.get("eval_precision", 0),
                "recall": eval_result.get("eval_recall", 0),
                "f1": eval_result.get("eval_f1", 0),
            }
        },
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
        "label_map": {"0": "not_related", "1": "disaster_related"},
    }
    metadata_path = output_dir / "roberta_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Metadata: {metadata_path}")

    print("\n" + "=" * 60)
    print("ROBERTA EGITIMI TAMAMLANDI")
    print("=" * 60)
    print(f"Model kaydedildi: {output_dir}")
    print("Inference icin: POST /t1/analyze with {\"text\": \"...\", \"use_roberta\": true}")
    return str(output_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RoBERTa disaster relevance fine-tuning")
    parser.add_argument("--output_dir", type=str, default=None, help="Model cikti dizini (varsayilan: models/roberta)")
    parser.add_argument("--epochs", type=int, default=3, help="Epoch sayisi")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH, help="Max sequence length")
    args = parser.parse_args()

    train_roberta(
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
    )
