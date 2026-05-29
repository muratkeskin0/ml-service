"""
T2 – Help Request and Humanitarian Category Detection
For disaster-related posts: classify help request and assign humanitarian category.
Output: discrete labels + class probabilities.
"""
import re
from typing import Tuple, List, Dict

# Humanitarian categories (discrete labels)
CATEGORIES = [
    "urgent_needs",
    "infrastructure_damage",
    "donations_volunteering",
    "other",
]

# Help request indicators (phrases and words)
HELP_REQUEST_PATTERNS = [
    r"\bneed\s+(help|assistance|support|aid|rescue|medicine|food|water|shelter)\b",
    r"\bplease\s+help\b",
    r"\blooking\s+for\s+(help|someone|family|people)\b",
    r"\brequest(ing)?\s+(help|assistance|aid|support)\b",
    r"\basking\s+for\s+help\b",
    r"\banyone\s+(can|please)\s+help\b",
    r"\b(urgent|emergency)\s+(need|help|request)\b",
    r"\b(stuck|trapped|stranded)\s+(need|please)\b",
    r"\b(can\s+someone|someone\s+please)\s+help\b",
    r"\bhelp\s+(us|me|them|needed)\b",
    r"\bwe\s+need\s+help\b",
    r"\byard[ıi]m\s+ed(in|er\s+misiniz)\b",
    r"\bihtiya[çc]\b",
    r"\bacil\s+yard[ıi]m\b",
    r"\bdonate\s+blood\b",
    r"\bneed\s+(shelter|evacuation|medical)\b",
    r"\bmissing\s+(person|people|family)\b",
    r"\bsearch(ing)?\s+for\s+(missing|family)\b",
]
HELP_REQUEST_KEYWORDS = [
    "need help", "please help", "urgent need", "requesting help",
    "asking for help", "help needed", "anyone help", "can someone help",
    "looking for help", "emergency need", "need assistance", "need rescue",
]

# Category keywords (English) – map to CATEGORIES
CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "urgent_needs": [
        "need help", "need rescue", "need medicine", "need food", "need water",
        "need shelter", "missing", "trapped", "stuck", "stranded", "evacuate",
        "injured", "dead", "casualties", "survivors need", "urgent", "emergency",
        "displaced", "evacuation", "missing person", "search for", "rescue needed",
    ],
    "infrastructure_damage": [
        "building collapse", "collapse", "damage", "destroyed", "flooded",
        "road blocked", "bridge", "infrastructure", "power outage", "no electricity",
        "water supply", "gas leak", "debris", "rubble", "structural",
    ],
    "donations_volunteering": [
        "donate", "donation", "volunteer", "fundraising", "relief fund",
        "how to help", "support relief", "aid organization", "red cross",
        "send aid", "volunteers needed", "raising money", "crowdfunding",
    ],
    "other": [
        "pray", "thoughts", "sympathy", "update", "news", "report", "situation",
        "caution", "advice", "warning", "information", "spread the word",
    ],
}


class T2HumanitarianClassifier:
    """
    T2: Help request detection + humanitarian category with probabilities.
    Rule-based first; can be replaced by a trained model later.
    """

    def __init__(self):
        self._help_compiled = [re.compile(p, re.I) for p in HELP_REQUEST_PATTERNS]

    def analyze(self, text: str) -> Dict:
        """
        Run T2 on a single text (assumed disaster-related).

        Returns:
            {
                "is_help_request": bool,
                "help_request_probability": float,
                "humanitarian_labels": list[str],  # one or more from CATEGORIES
                "category_probabilities": dict[str, float],
            }
        """
        if not text or not text.strip():
            return self._empty_result()

        text_lower = text.lower().strip()
        is_help, help_prob = self._classify_help_request(text_lower)
        labels, probs = self._classify_categories(text_lower)

        return {
            "is_help_request": is_help,
            "help_request_probability": round(help_prob, 4),
            "humanitarian_labels": labels,
            "category_probabilities": {k: round(v, 4) for k, v in probs.items()},
        }

    def _empty_result(self) -> Dict:
        probs = {c: 0.0 for c in CATEGORIES}
        probs["other"] = 1.0
        return {
            "is_help_request": False,
            "help_request_probability": 0.0,
            "humanitarian_labels": ["other"],
            "category_probabilities": probs,
        }

    def _classify_help_request(self, text_lower: str) -> Tuple[bool, float]:
        """Help request: binary + probability in [0, 1]."""
        score = 0.0
        for pat in self._help_compiled:
            if pat.search(text_lower):
                score += 0.35
                break
        for kw in HELP_REQUEST_KEYWORDS:
            if kw in text_lower:
                score += 0.25
                break
        # Length-normalized: short urgent phrases count more
        if "help" in text_lower or "need" in text_lower:
            score += 0.15
        if "please" in text_lower or "urgent" in text_lower:
            score += 0.1
        prob = min(1.0, score)
        return prob >= 0.35, prob

    def _classify_categories(self, text_lower: str) -> Tuple[List[str], Dict[str, float]]:
        """Multi-label categories with probabilities (sum to 1)."""
        scores = {c: 0.0 for c in CATEGORIES}
        for cat, keywords in CATEGORY_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    scores[cat] += 1.0
        # If no match, default to "other"
        total = sum(scores.values())
        if total <= 0:
            scores["other"] = 1.0
            total = 1.0
        probs = {c: scores[c] / total for c in CATEGORIES}
        # Labels: all categories above a threshold (e.g. 0.2), or top-1
        threshold = 0.2
        labels = [c for c in CATEGORIES if probs[c] >= threshold]
        if not labels:
            labels = [max(CATEGORIES, key=lambda x: probs[x])]
        return labels, probs


# Singleton for FastAPI
_t2_classifier: T2HumanitarianClassifier = None


def get_t2_classifier() -> T2HumanitarianClassifier:
    global _t2_classifier
    if _t2_classifier is None:
        _t2_classifier = T2HumanitarianClassifier()
    return _t2_classifier
