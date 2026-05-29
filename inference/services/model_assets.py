"""
Model asset helpers: download inference weights from Hugging Face Hub (free, public)
and FastText LID from Meta's official CDN.

Environment:
  ML_MODELS_HF_REPO  Hugging Face model repo id (default: MuratKeskin0/smart-disaster-hub-ml)
"""
from __future__ import annotations

import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

DEFAULT_HF_REPO = "MuratKeskin0/smart-disaster-hub-ml"
FASTTEXT_LID_FTZ_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"

LOGISTIC_REGRESSION_FILES = (
    "model.pkl",
    "vectorizer.pkl",
    "char_vectorizer.pkl",
    "feature_extractor.pkl",
    "model_metadata.json",
)
LOGISTIC_REGRESSION_REQUIRED = ("model.pkl", "vectorizer.pkl")


def get_hf_repo_id() -> str:
    return os.environ.get("ML_MODELS_HF_REPO", DEFAULT_HF_REPO).strip()


def _services_dir() -> Path:
    return Path(__file__).resolve().parent


def default_text_analyzer_models_dir() -> Path:
    return _services_dir() / "text_analyzer" / "models"


def default_language_detector_models_dir() -> Path:
    return _services_dir() / "language_detector" / "models"


def _missing_files(model_dir: Path, filenames: Iterable[str]) -> list[str]:
    return [name for name in filenames if not (model_dir / name).exists()]


def _download_hf_file(repo_id: str, filename: str, destination: Path) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download models. "
            "Install with: pip install huggingface_hub"
        ) from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(destination.parent),
    )
    if not destination.exists():
        raise FileNotFoundError(f"Download failed: {filename} from {repo_id}")


def ensure_logistic_regression_models(
    model_dir: Optional[Path] = None,
    repo_id: Optional[str] = None,
    required: bool = True,
) -> bool:
    """Download Logistic Regression artifacts if missing. Returns True if LR is ready."""
    model_dir = model_dir or default_text_analyzer_models_dir()
    repo_id = repo_id or get_hf_repo_id()
    missing_required = _missing_files(model_dir, LOGISTIC_REGRESSION_REQUIRED)
    if not missing_required:
        return True

    logger.info(
        "Logistic Regression model files missing in %s — downloading from %s",
        model_dir,
        repo_id,
    )
    errors: list[str] = []
    for filename in LOGISTIC_REGRESSION_FILES:
        destination = model_dir / filename
        if destination.exists():
            continue
        try:
            _download_hf_file(repo_id, filename, destination)
            logger.info("Downloaded %s", filename)
        except Exception as exc:
            if filename in LOGISTIC_REGRESSION_REQUIRED:
                errors.append(f"{filename}: {exc}")
            else:
                logger.debug("Optional file not on Hub: %s (%s)", filename, exc)

    if not _missing_files(model_dir, LOGISTIC_REGRESSION_REQUIRED):
        return True

    if not required:
        logger.info("Logistic Regression weights unavailable; RoBERTa-only mode may be used")
        return False

    still_missing = _missing_files(model_dir, LOGISTIC_REGRESSION_REQUIRED)
    setup_hint = (
        "See SETUP.md or run: python scripts/download_models.py --roberta"
    )
    detail = "; ".join(errors) if errors else ", ".join(still_missing)
    raise FileNotFoundError(
        f"Required model files not found: {', '.join(still_missing)}. {detail}. {setup_hint}"
    )


def roberta_weights_available(model_dir: Optional[Path] = None) -> bool:
    model_dir = model_dir or default_text_analyzer_models_dir()
    roberta_dir = model_dir / "roberta"
    return (roberta_dir / "config.json").exists() and (
        (roberta_dir / "pytorch_model.bin").exists()
        or (roberta_dir / "model.safetensors").exists()
    )


def ensure_fasttext_model(models_dir: Optional[Path] = None) -> Optional[Path]:
    """Download FastText LID (.ftz) from Meta CDN if missing."""
    models_dir = models_dir or default_language_detector_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)

    for name in ("lid.176.ftz", "lid.176.bin"):
        path = models_dir / name
        if path.exists():
            return path

    destination = models_dir / "lid.176.ftz"
    logger.info("FastText LID model missing — downloading from Meta CDN")
    try:
        urllib.request.urlretrieve(FASTTEXT_LID_FTZ_URL, destination)
    except urllib.error.URLError as exc:
        logger.warning("FastText download failed (%s); heuristic language detection will be used", exc)
        return None

    logger.info("FastText LID model saved to %s", destination)
    return destination


def ensure_roberta_models(
    model_dir: Optional[Path] = None,
    repo_id: Optional[str] = None,
) -> bool:
    """Download optional RoBERTa folder from Hugging Face if config is missing."""
    model_dir = model_dir or default_text_analyzer_models_dir()
    roberta_dir = model_dir / "roberta"
    if (roberta_dir / "config.json").exists() and (
        (roberta_dir / "pytorch_model.bin").exists()
        or (roberta_dir / "model.safetensors").exists()
    ):
        return True

    repo_id = repo_id or get_hf_repo_id()
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.warning("huggingface_hub not installed; skipping RoBERTa download")
        return False

    try:
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=["roberta/**"],
            local_dir=str(model_dir),
        )
    except Exception as exc:
        logger.debug("RoBERTa snapshot not available on Hub: %s", exc)
        return False

    return (roberta_dir / "config.json").exists()


def ensure_all_models(include_roberta: bool = True) -> None:
    """Download assets for a fresh clone (RoBERTa-first when LR weights are absent)."""
    lr_ready = ensure_logistic_regression_models(required=False)
    ensure_fasttext_model()
    roberta_ready = ensure_roberta_models()
    if not lr_ready and not roberta_ready:
        raise FileNotFoundError(
            "No inference weights found. Run: python scripts/download_models.py --roberta"
        )
