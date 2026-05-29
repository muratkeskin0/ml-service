#!/usr/bin/env python3
"""
Upload local model files to a free public Hugging Face model repo.

Uploads every file that exists locally; missing files are skipped.

Usage:
  hf auth login
  python scripts/upload_models_to_hf.py --models-dir path/to/models
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = ROOT / "inference" / "services" / "text_analyzer" / "models"
DEFAULT_REPO = "MuratKeskin0/smart-disaster-hub-ml"

LOGISTIC_FILES = (
    "model.pkl",
    "vectorizer.pkl",
    "char_vectorizer.pkl",
    "feature_extractor.pkl",
    "model_metadata.json",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload ML models to Hugging Face Hub (free)")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Target HF model repo id")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--private", action="store_true", help="Create private repo (free tier limit applies)")
    args = parser.parse_args()

    models_dir: Path = args.models_dir
    if not models_dir.exists():
        print(f"ERROR: models directory not found: {models_dir}", file=sys.stderr)
        return 1

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: pip install huggingface_hub", file=sys.stderr)
        return 1

    api = HfApi()
    api.create_repo(
        repo_id=args.repo,
        repo_type="model",
        exist_ok=True,
        private=args.private,
    )

    uploaded = 0
    for name in LOGISTIC_FILES:
        path = models_dir / name
        if path.exists():
            print(f"Uploading {name} ...")
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=name,
                repo_id=args.repo,
                repo_type="model",
            )
            uploaded += 1
        else:
            print(f"Skipping {name} (not found locally)")

    roberta_dir = models_dir / "roberta"
    if roberta_dir.is_dir() and any(roberta_dir.iterdir()):
        print("Uploading roberta/ folder ...")
        api.upload_folder(
            folder_path=str(roberta_dir),
            path_in_repo="roberta",
            repo_id=args.repo,
            repo_type="model",
        )
        uploaded += 1

    if uploaded == 0:
        print("ERROR: no model files found to upload.", file=sys.stderr)
        return 1

    print(f"\nDone ({uploaded} item(s)). Repo: https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
