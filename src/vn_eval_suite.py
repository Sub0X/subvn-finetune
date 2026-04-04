from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


PLACEHOLDER_RE = re.compile(r"%name|%nick|%fp|<NAME>|<NICK>|<FP>")
SPEAKER_TAG_RE = re.compile(r"\[[^\]\n]{1,80}\]")
HONORIFIC_RE = re.compile(r"\b[A-Za-z][A-Za-z' -]{0,40}-(san|chan|kun|sama|senpai|sensei|dono)\b", re.IGNORECASE)
INLINE_CODE_RE = re.compile(r"`[^`\n]+`|<\|[^>]+\|>")
VNTL_PAIR_RE = re.compile(r"<<JAPANESE>>(.*?)<<ENGLISH>>(.*?)(?=<<JAPANESE>>|$)", re.DOTALL)


def _safe_text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _token_set(pattern: re.Pattern[str], text: str) -> set[str]:
    return set(pattern.findall(_safe_text(text)))


def extract_vntl_pair(text: str, ignore_loss: list[int] | None = None) -> tuple[str, str] | None:
    matches = list(VNTL_PAIR_RE.finditer(_safe_text(text)))
    if not matches:
        return None

    ignored = {int(x) for x in (ignore_loss or []) if isinstance(x, (int, float))}
    ja_segments: list[str] = []
    en_segments: list[str] = []
    for idx, match in enumerate(matches):
        if idx in ignored:
            continue
        ja = match.group(1).strip()
        en = match.group(2).strip()
        if ja and en:
            ja_segments.append(ja)
            en_segments.append(en)

    if not ja_segments or not en_segments:
        return None
    return "\n\n".join(ja_segments), "\n\n".join(en_segments)


@dataclass
class TargetedCheckResult:
    placeholder_exact: bool
    speaker_tag_subset: bool
    honorific_alignment: bool
    inline_markup_subset: bool
    length_ratio: float
    suspiciously_short: bool


def run_targeted_checks(source_ja: str, reference_en: str, prediction_en: str) -> TargetedCheckResult:
    ref_placeholders = _token_set(PLACEHOLDER_RE, reference_en) | _token_set(PLACEHOLDER_RE, source_ja)
    pred_placeholders = _token_set(PLACEHOLDER_RE, prediction_en)

    ref_speakers = _token_set(SPEAKER_TAG_RE, reference_en) | _token_set(SPEAKER_TAG_RE, source_ja)
    pred_speakers = _token_set(SPEAKER_TAG_RE, prediction_en)

    ref_markup = _token_set(INLINE_CODE_RE, reference_en) | _token_set(INLINE_CODE_RE, source_ja)
    pred_markup = _token_set(INLINE_CODE_RE, prediction_en)

    ref_honorifics = _token_set(HONORIFIC_RE, reference_en)
    pred_honorifics = _token_set(HONORIFIC_RE, prediction_en)

    ref_len = max(1, len(reference_en.strip()))
    pred_len = len(prediction_en.strip())
    length_ratio = pred_len / ref_len

    return TargetedCheckResult(
        placeholder_exact=pred_placeholders == ref_placeholders,
        speaker_tag_subset=ref_speakers.issubset(pred_speakers) if ref_speakers else True,
        honorific_alignment=(not ref_honorifics and not pred_honorifics) or (ref_honorifics == pred_honorifics),
        inline_markup_subset=ref_markup.issubset(pred_markup) if ref_markup else True,
        length_ratio=length_ratio,
        suspiciously_short=length_ratio < 0.55,
    )


def compute_reference_metrics(predictions: list[str], references: list[str]) -> dict[str, float | None]:
    try:
        import sacrebleu
    except Exception:
        return {"chrf2": None, "ter": None}

    chrf = sacrebleu.corpus_chrf(predictions, [references], word_order=2)
    ter = sacrebleu.corpus_ter(predictions, [references])
    return {"chrf2": float(chrf.score), "ter": float(ter.score)}


def compute_comet_metrics(
    sources: list[str],
    predictions: list[str],
    references: list[str],
    model_name: str = "Unbabel/wmt22-comet-da",
) -> dict[str, float | None]:
    try:
        from comet import download_model, load_from_checkpoint
    except Exception:
        return {"comet": None}

    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)
    rows = [{"src": s, "mt": p, "ref": r} for s, p, r in zip(sources, predictions, references)]
    result = model.predict(rows, batch_size=4, gpus=1 if model.device.type == "cuda" else 0)
    return {"comet": float(result.system_score)}


def summarize_targeted_results(results: Iterable[TargetedCheckResult]) -> dict[str, float]:
    rows = list(results)
    if not rows:
        return {
            "placeholder_exact_rate": math.nan,
            "speaker_tag_subset_rate": math.nan,
            "honorific_alignment_rate": math.nan,
            "inline_markup_subset_rate": math.nan,
            "suspiciously_short_rate": math.nan,
        }

    def rate(attr: str) -> float:
        return sum(1 for row in rows if getattr(row, attr)) / len(rows)

    return {
        "placeholder_exact_rate": rate("placeholder_exact"),
        "speaker_tag_subset_rate": rate("speaker_tag_subset"),
        "honorific_alignment_rate": rate("honorific_alignment"),
        "inline_markup_subset_rate": rate("inline_markup_subset"),
        "suspiciously_short_rate": rate("suspiciously_short"),
    }


def load_prediction_rows(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def evaluate_prediction_rows(rows: list[dict], include_comet: bool = False) -> dict:
    sources = [_safe_text(row.get("source_ja")) for row in rows]
    references = [_safe_text(row.get("reference_en")) for row in rows]
    predictions = [_safe_text(row.get("prediction_en")) for row in rows]

    targeted = [
        run_targeted_checks(src, ref, pred)
        for src, ref, pred in zip(sources, references, predictions)
    ]

    summary = {
        "rows": len(rows),
        "reference_metrics": compute_reference_metrics(predictions, references),
        "targeted_checks": summarize_targeted_results(targeted),
        "examples": [
            {
                "id": row.get("id"),
                "track": row.get("track"),
                "content_bucket": row.get("content_bucket"),
                **asdict(result),
            }
            for row, result in zip(rows, targeted)
        ],
    }

    if include_comet:
        summary["comet_metrics"] = compute_comet_metrics(sources, predictions, references)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VN translation predictions with reference and targeted checks.")
    parser.add_argument("--predictions", required=True, help="JSONL file with source_ja/reference_en/prediction_en fields.")
    parser.add_argument("--output", required=True, help="Where to write the evaluation summary JSON.")
    parser.add_argument("--include-comet", action="store_true", help="Attempt COMET scoring if the package is installed.")
    args = parser.parse_args()

    rows = load_prediction_rows(args.predictions)
    summary = evaluate_prediction_rows(rows, include_comet=args.include_comet)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote evaluation summary to {output_path}")


if __name__ == "__main__":
    main()
