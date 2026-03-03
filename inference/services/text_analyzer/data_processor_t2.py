"""
T2 etiketleri ile veri yükleme: disaster + help_request + humanitarian_category.
HumAID ve archive'taki class_label/label -> urgent_needs, infrastructure_damage, donations_volunteering, other.
"""
import csv
import re
from pathlib import Path
from typing import List, Tuple

# T2 kategori indeksleri (T2 çıktısı ile aynı)
CATEGORY_URGENT_NEEDS = 0
CATEGORY_INFRASTRUCTURE = 1
CATEGORY_DONATIONS = 2
CATEGORY_OTHER = 3
CATEGORY_NAMES = ["urgent_needs", "infrastructure_damage", "donations_volunteering", "other"]


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _map_raw_label_to_t2(label: str) -> Tuple[int, int, int]:
    """
    Raw label (HumAID/archive) -> (disaster, help_request, category_id).
    Returns (1, 0, 3) for not_related so we still have a valid category.
    """
    l = (label or "").strip().lower()
    if not l or "not_related" in l or "not_humanitarian" in l or "not_informative" in l:
        return (0, 0, CATEGORY_OTHER)

    # Help request: acil yardım/arama ihtiyacı
    help_request_labels = [
        "requests_or_needs",
        "missing_or_found_people",
        "missing_trapped_or_found_people",
        "affected_individual",
    ]
    is_help = any(h in l for h in help_request_labels)

    # Kategori (donation/volunteer önce, sonra infrastructure, sonra urgent)
    if "donation" in l or "volunteer" in l or "rescue_volunteering" in l:
        return (1, 0, CATEGORY_DONATIONS)
    if "infrastructure" in l or "utility" in l or "utilities" in l:
        return (1, 0, CATEGORY_INFRASTRUCTURE)
    if "injured" in l or "dead" in l or "missing" in l or "displaced" in l or "evacuation" in l or "affected_individual" in l or "requests_or_needs" in l or "rescue" in l:
        return (1, 1 if is_help else 0, CATEGORY_URGENT_NEEDS)
    # other_relevant_information, caution_and_advice, sympathy_and_support, other_useful_information
    return (1, 0, CATEGORY_OTHER)


def load_humaid_t2(data_dir: Path) -> List[Tuple[str, int, int, int, str]]:
    """(text, disaster, help_request, category_id, source)"""
    out = []
    humaid_dir = data_dir / "HumAID_data_events_set1_47K" / "events_set1"
    if not humaid_dir.exists():
        return out
    for event_dir in humaid_dir.iterdir():
        if not event_dir.is_dir():
            continue
        name = event_dir.name
        for split in ["train", "dev", "test"]:
            f = event_dir / f"{name}_{split}.tsv"
            if not f.exists():
                continue
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    r = csv.DictReader(fp, delimiter="\t")
                    for row in r:
                        text = (row.get("tweet_text") or "").strip()
                        cl = (row.get("class_label") or "").strip()
                        if len(text) < 3:
                            continue
                        text = _clean_text(text)
                        if len(text) < 3:
                            continue
                        d, h, c = _map_raw_label_to_t2(cl)
                        out.append((text, d, h, c, f"humaid_{name}_{split}"))
            except Exception as e:
                print(f"  [WARN] {f.name}: {e}")
    return out


def load_archive_t2(data_dir: Path) -> List[Tuple[str, int, int, int, str]]:
    """(text, disaster, help_request, category_id, source) - archive (1) TSV"""
    out = []
    archive_dir = data_dir / "archive (1)"
    if not archive_dir.exists():
        return out
    for tsv in archive_dir.glob("*.tsv"):
        try:
            with open(tsv, "r", encoding="utf-8") as fp:
                r = csv.DictReader(fp, delimiter="\t")
                event = tsv.stem.replace("_CF_labeled_data", "").replace("_cl_labeled_data", "").replace("_en_CF_labeled_data", "")
                for row in r:
                    text = (row.get("tweet_text") or "").strip()
                    label = (row.get("label") or "").strip()
                    if len(text) < 3:
                        continue
                    text = _clean_text(text)
                    if len(text) < 3:
                        continue
                    d, h, c = _map_raw_label_to_t2(label)
                    out.append((text, d, h, c, f"archive_{event}"))
        except Exception as e:
            print(f"  [WARN] {tsv.name}: {e}")
    return out


def load_crisis_benchmarks_t2(data_dir: Path) -> List[Tuple[str, int, int, int, str]]:
    """(text, disaster, help_request, category_id, source) - all_data_en"""
    out = []
    bench_dir = data_dir / "crisis_datasets_benchmarks_v1.0" / "data" / "all_data_en"
    if not bench_dir.exists():
        return out
    for tsv in bench_dir.glob("*.tsv"):
        try:
            with open(tsv, "r", encoding="utf-8") as fp:
                r = csv.DictReader(fp, delimiter="\t")
                for row in r:
                    if (row.get("lang") or "").strip().lower() != "en":
                        continue
                    text = (row.get("text") or "").strip()
                    cl = (row.get("class_label") or "").strip()
                    if len(text) < 3:
                        continue
                    text = _clean_text(text)
                    if len(text) < 3:
                        continue
                    d, h, c = _map_raw_label_to_t2(cl)
                    out.append((text, d, h, c, "crisis_benchmarks"))
        except Exception as e:
            print(f"  [WARN] {tsv.name}: {e}")
    return out


def process_all_datasets_t2(data_dir: Path = None) -> Tuple[List[str], List[int], List[int], List[int]]:
    """
    Tüm veriyi T2 etiketleriyle yükle.
    Returns: (texts, disaster_labels, help_request_labels, category_ids)
    T2 etiketi olmayan kaynaklar için help_request=0, category=OTHER kullanılır.
    """
    if data_dir is None:
        data_dir = Path(__file__).parent / "data"
    data_dir = Path(data_dir)

    texts, disaster, help_req, categories = [], [], [], []

    # T2 etiketli kaynaklar
    for text, d, h, c, _ in load_humaid_t2(data_dir) + load_archive_t2(data_dir) + load_crisis_benchmarks_t2(data_dir):
        texts.append(text)
        disaster.append(d)
        help_req.append(h)
        categories.append(c)

    # T1-only kaynaklar (T2 etiketi yok -> help=0, category=other)
    try:
        from data_processor import DataProcessor
        proc = DataProcessor(str(data_dir))
        t1_texts, t1_labels, t1_sources = proc.process_all_datasets()
        t2_source_prefixes = ("humaid_", "archive_", "crisis_benchmarks")
        for t, d, src in zip(t1_texts, t1_labels, t1_sources):
            if src.startswith(t2_source_prefixes):
                continue
            if len(t.strip()) < 10:
                continue
            texts.append(t)
            disaster.append(d)
            help_req.append(0)
            categories.append(CATEGORY_OTHER)
    except Exception as e:
        print(f"  [WARN] T1-only data: {e}")

    print(f"  [T2] Toplam {len(texts)} kayit (disaster={sum(disaster)}, help_req={sum(help_req)}, categories: {sum(1 for c in categories if c==0)} urgent, {sum(1 for c in categories if c==1)} infra, {sum(1 for c in categories if c==2)} don, {sum(1 for c in categories if c==3)} other)")
    return texts, disaster, help_req, categories
