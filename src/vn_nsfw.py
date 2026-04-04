from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable

def _iter_candidate_paths(extra_paths: Iterable[str | Path] | None = None) -> list[Path]:
    paths: list[Path] = []
    if extra_paths:
        for value in extra_paths:
            if value:
                paths.append(Path(value))
    return paths


@lru_cache(maxsize=8)
def load_keyword_scores(path_str: str | None = None) -> dict[str, int]:
    candidates = _iter_candidate_paths([path_str] if path_str else None)
    for path in candidates:
        if not path.exists():
            continue
        scores: dict[str, int] = {}
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "\t" in line:
                    term, score_text = line.split("\t", 1)
                else:
                    term, score_text = line, "1"
                term = term.strip()
                if not term:
                    continue
                try:
                    score = int(score_text.strip())
                except ValueError:
                    score = 1
                scores[term] = score
        if scores:
            return scores
    return {}


def nsfw_score(text: str, keyword_scores: dict[str, int] | None = None) -> int:
    if not isinstance(text, str):
        return 0
    keyword_scores = keyword_scores or load_keyword_scores()
    lowered = text.lower()
    score = 0

    for term, weight in keyword_scores.items():
        haystack = lowered if term.isascii() else text
        needle = term.lower() if term.isascii() else term
        if needle in haystack:
            score += weight

    # Generic censored-token pattern often used in explicit VN lines.
    if re.search(r"[\u3041-\u3093\u30a1-\u30f3\u4e00-\u9fffa-zA-Z]+\u25cf", text):
        score += 3

    # Moaning-heavy orthography in JP lines can be a weak signal.
    if re.search(r"[\u3041\u3045\u3049\u30a1\u30a5\u30a9]{2,}.*[\u30c3\u3063]{1,}", text):
        score += 1

    # Combination signal often found in explicit translated lines.
    if ("womb" in lowered or "子宮" in text) and ("deep" in lowered or "inside" in lowered or "挿入" in text):
        score += 2

    return score


def is_nsfw_text(text: str, level: str = "explicit_only", keyword_scores: dict[str, int] | None = None) -> bool:
    score = nsfw_score(text, keyword_scores=keyword_scores)
    if level == "minimal":
        return score >= 5
    if level == "moderate":
        return score >= 1
    return score >= 2


def is_nsfw_row(row: dict, level: str = "explicit_only", keyword_scores: dict[str, int] | None = None) -> bool:
    if not isinstance(row, dict):
        return False

    text = row.get("text", "")
    if is_nsfw_text(text, level=level, keyword_scores=keyword_scores):
        return True

    conversations = row.get("conversations", None)
    if isinstance(conversations, list):
        for turn in conversations:
            if not isinstance(turn, dict):
                continue
            content = turn.get("content", None)
            if isinstance(content, str) and is_nsfw_text(content, level=level, keyword_scores=keyword_scores):
                return True
            if isinstance(content, list):
                for chunk in content:
                    if isinstance(chunk, dict) and is_nsfw_text(chunk.get("text", ""), level=level, keyword_scores=keyword_scores):
                        return True
    return False


def classify_content_bucket(text: str, keyword_scores: dict[str, int] | None = None) -> str:
    return "explicit" if is_nsfw_text(text, level="moderate", keyword_scores=keyword_scores) else "general"
