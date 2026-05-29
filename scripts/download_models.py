#!/usr/bin/env python3
"""Download ML inference assets for a fresh clone."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "inference"))

from services.model_assets import (  # noqa: E402
    ensure_all_models,
    ensure_fasttext_model,
    ensure_logistic_regression_models,
    ensure_roberta_models,
    get_hf_repo_id,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Smart Disaster Hub ML model files")
    parser.add_argument(
        "--repo",
        default=None,
        help=f"Hugging Face repo id (default: env ML_MODELS_HF_REPO or {get_hf_repo_id()})",
    )
    parser.add_argument("--roberta", action="store_true", help="Also download optional RoBERTa weights")
    parser.add_argument("--fasttext-only", action="store_true", help="Only download FastText LID model")
    args = parser.parse_args()

    repo_id = args.repo
    try:
        if args.fasttext_only:
            path = ensure_fasttext_model()
            print(f"FastText: {path or 'download skipped (will use heuristics)'}")
            return 0

        if args.roberta:
            ensure_all_models(include_roberta=True)
        else:
            ensure_logistic_regression_models(repo_id=repo_id)
            ensure_fasttext_model()

        if args.roberta:
            ok = ensure_roberta_models(repo_id=repo_id)
            print(f"RoBERTa: {'ready' if ok else 'not found on Hub (optional)'}")

        print(f"Models ready. Hugging Face repo: {repo_id or get_hf_repo_id()}")
        return 0
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
