"""
RoBERTa Multi-Task: T1 (disaster relevance) + T2 (help request, humanitarian category)
Tek model, tek forward pass. Colab'da train_roberta.py yerine bunu çalıştırabilirsin.
"""
import sys
import os
import codecs
import json
from pathlib import Path
from collections import Counter

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) != os.getcwd():
    os.chdir(_SCRIPT_DIR)
    sys.path.insert(0, str(_SCRIPT_DIR))

if sys.platform == "win32":
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data_processor_t2 import process_all_datasets_t2, CATEGORY_NAMES, CATEGORY_OTHER
from model_multitask_roberta import MultiTaskRobertaForClassification

MAX_LENGTH = 512
MODEL_NAME = "roberta-base"
ROBERTA_SUBDIR = "roberta"
MIN_TEXT_LENGTH = 10


class MultiTaskDataset(Dataset):
    def __init__(self, texts, labels_d, labels_h, labels_c, tokenizer, max_length=MAX_LENGTH):
        self.texts = list(texts)
        self.ld = list(labels_d)
        self.lh = list(labels_h)
        self.lc = list(labels_c)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tokenizer(
            self.texts[i],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels_disaster": torch.tensor(self.ld[i], dtype=torch.long),
            "labels_help": torch.tensor(self.lh[i], dtype=torch.long),
            "labels_category": torch.tensor(self.lc[i], dtype=torch.long),
        }


def train_multitask(
    output_dir=None,
    num_epochs=3,
    batch_size=16,
    lr=2e-5,
    warmup_ratio=0.1,
    max_length=MAX_LENGTH,
):
    print("=" * 60)
    print("ROBERTA MULTI-TASK (T1 + T2)")
    print("=" * 60)

    script_dir = Path(__file__).parent
    output_dir = Path(output_dir or script_dir / "models" / ROBERTA_SUBDIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = script_dir / "data"

    print("\n[1/6] T2 etiketli veri yukleniyor...")
    texts, labels_d, labels_h, labels_c = process_all_datasets_t2(data_dir)
    if len(texts) == 0:
        raise RuntimeError("Veri yuklenemedi!")

    print("\n[2/6] Filtreleme ve split...")
    filtered = [(t, d, h, c) for t, d, h, c in zip(texts, labels_d, labels_h, labels_c) if len(t.strip()) >= MIN_TEXT_LENGTH]
    texts, labels_d, labels_h, labels_c = zip(*filtered) if filtered else ([], [], [], [])
    texts, labels_d, labels_h, labels_c = list(texts), list(labels_d), list(labels_h), list(labels_c)
    X_train, X_val, yd_train, yd_val, yh_train, yh_val, yc_train, yc_val = train_test_split(
        texts, labels_d, labels_h, labels_c, test_size=0.15, random_state=42, stratify=labels_d
    )
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}")

    print("\n[3/6] Tokenizer ve model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = MultiTaskRobertaForClassification(base_model_name=MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_ds = MultiTaskDataset(X_train, yd_train, yh_train, yc_train, tokenizer, max_length)
    val_ds = MultiTaskDataset(X_val, yd_val, yh_val, yc_val, tokenizer, max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0)

    total_steps = len(train_loader) * num_epochs
    warmup = int(warmup_ratio * total_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup, num_training_steps=total_steps)

    print("\n[4/6] Egitim...")
    best_val_f1 = 0.0
    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            for k, v in batch.items():
                batch[k] = v.to(device)
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels_disaster=batch["labels_disaster"],
                labels_help=batch["labels_help"],
                labels_category=batch["labels_category"],
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if (step + 1) % 100 == 0:
                print(f"  Epoch {epoch+1} step {step+1} loss {out.loss.item():.4f}")

        model.eval()
        acc_d, acc_h, acc_c = 0.0, 0.0, 0.0
        f1_d = 0.0
        with torch.no_grad():
            for batch in val_loader:
                for k, v in batch.items():
                    batch[k] = v.to(device)
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                logits_d, logits_h, logits_c = out.logits
                pred_d = logits_d.argmax(1).cpu().numpy()
                pred_h = logits_h.argmax(1).cpu().numpy()
                pred_c = logits_c.argmax(1).cpu().numpy()
                acc_d += (pred_d == batch["labels_disaster"].cpu().numpy()).sum()
                acc_h += (pred_h == batch["labels_help"].cpu().numpy()).sum()
                acc_c += (pred_c == batch["labels_category"].cpu().numpy()).sum()
                f1_d += f1_score(batch["labels_disaster"].cpu().numpy(), pred_d, zero_division=0) * len(pred_d)
        n_val = len(X_val)
        acc_d, acc_h, acc_c = acc_d / n_val, acc_h / n_val, acc_c / n_val
        f1_d = f1_d / n_val
        print(f"  Val Epoch {epoch+1} acc_d={acc_d:.4f} acc_h={acc_h:.4f} acc_c={acc_c:.4f} f1_d={f1_d:.4f}")
        if f1_d > best_val_f1:
            best_val_f1 = f1_d
            torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
            print(f"  [OK] Best model saved (f1_d={f1_d:.4f})")

    print("\n[5/6] Model ve tokenizer kaydediliyor...")
    model.load_state_dict(torch.load(output_dir / "pytorch_model.bin", map_location=device))
    # Tam model state_dict (RoBERTa + 3 head) + tokenizer
    torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
    tokenizer.save_pretrained(output_dir)
    # Config: HuggingFace RoBERTa config + multitask alanları (inference için)
    from transformers import RobertaConfig
    roberta_config = RobertaConfig.from_pretrained(MODEL_NAME)
    config_dict = roberta_config.to_dict()
    config_dict["multitask"] = True
    config_dict["num_labels"] = 2
    config_dict["num_help"] = 2
    config_dict["num_category"] = 4
    config_dict["category_names"] = CATEGORY_NAMES
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    metadata = {
        "model_name": MODEL_NAME,
        "multitask": True,
        "tasks": ["disaster_relevance", "help_request", "humanitarian_category"],
        "category_names": CATEGORY_NAMES,
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "best_val_f1_disaster": best_val_f1,
    }
    with open(output_dir / "roberta_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  [OK] {output_dir}")

    print("\n" + "=" * 60)
    print("MULTI-TASK EGITIM TAMAMLANDI")
    print("=" * 60)
    print("Inference: /analyze use_roberta=true ile T1+T2 tek modelden döner.")
    return str(output_dir)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_length", type=int, default=MAX_LENGTH)
    args = p.parse_args()
    train_multitask(
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
    )
