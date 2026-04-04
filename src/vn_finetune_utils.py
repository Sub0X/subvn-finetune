from __future__ import annotations

import json
import math
import random
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping

from vn_nsfw import classify_content_bucket


def has_placeholders(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return any(token in text for token in ("%name", "%nick", "%fp", "<NAME>", "<NICK>", "<FP>"))


def has_censored_token(text: str) -> bool:
    return isinstance(text, str) and "\u25cf" in text


def count_vntl_pairs(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return text.count("<<JAPANESE>>")


def _pair_bucket(pair_count: int) -> str:
    if pair_count <= 8:
        return "pairs_08"
    if pair_count <= 12:
        return "pairs_12"
    if pair_count <= 16:
        return "pairs_16"
    return "pairs_17_plus"


def vntl_stratify_key(row: Mapping[str, Any]) -> str:
    text = str(row.get("text", "") or "")
    parts = [
        classify_content_bucket(text),
        "placeholder" if has_placeholders(text) else "plain",
        "censored" if has_censored_token(text) else "uncensored",
        _pair_bucket(count_vntl_pairs(text)),
    ]
    return "|".join(parts)


def shisa_stratify_key(row: Mapping[str, Any]) -> str:
    conversations = row.get("conversations", []) or []
    first_turn = ""
    if conversations:
        try:
            first_turn = str(conversations[0].get("value", "") or "")
        except Exception:
            first_turn = str(conversations[0] or "")
    has_jp = any(("\u3040" <= ch <= "\u30ff") or ("\u4e00" <= ch <= "\u9fff") for ch in first_turn)
    source_model = str(row.get("source_model", "") or "unknown")
    return f"{'jp' if has_jp else 'non_jp'}|{source_model}"


def stratified_select(dataset: Any, sample_size: int, seed: int, key_fn: Callable[[Mapping[str, Any]], str]) -> Any:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    total_rows = len(dataset)
    if sample_size >= total_rows:
        return dataset.shuffle(seed=seed)

    groups: dict[str, list[int]] = defaultdict(list)
    for idx in range(total_rows):
        groups[key_fn(dataset[idx])].append(idx)

    rng = random.Random(seed)
    for indices in groups.values():
        rng.shuffle(indices)

    allocations: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    remaining = sample_size
    group_items = list(groups.items())

    for group_key, indices in group_items:
        exact = (len(indices) / total_rows) * sample_size
        base = min(len(indices), int(math.floor(exact)))
        if base == 0 and remaining > 0:
            base = 1
        allocations[group_key] = base
        remaining -= base
        remainders.append((exact - math.floor(exact), group_key))

    if remaining < 0:
        for _, group_key in sorted(remainders):
            if remaining == 0:
                break
            if allocations[group_key] > 1:
                allocations[group_key] -= 1
                remaining += 1

    for _, group_key in sorted(remainders, reverse=True):
        if remaining <= 0:
            break
        capacity = len(groups[group_key]) - allocations[group_key]
        if capacity <= 0:
            continue
        allocations[group_key] += 1
        remaining -= 1

    selected: list[int] = []
    for group_key, indices in group_items:
        selected.extend(indices[: allocations[group_key]])

    if len(selected) < sample_size:
        unused = [idx for indices in groups.values() for idx in indices if idx not in selected]
        rng.shuffle(unused)
        selected.extend(unused[: sample_size - len(selected)])

    rng.shuffle(selected)
    selected = selected[:sample_size]
    selected.sort()
    return dataset.select(selected)


def split_dataset_by_content(dataset: Any) -> OrderedDict[str, Any]:
    buckets: OrderedDict[str, Any] = OrderedDict()
    general = dataset.filter(lambda row: classify_content_bucket(row.get("text", "")) == "general")
    explicit = dataset.filter(lambda row: classify_content_bucket(row.get("text", "")) == "explicit")
    if len(general) > 0:
        buckets["general"] = general
    if len(explicit) > 0:
        buckets["explicit"] = explicit
    return buckets


def build_eval_tracks(track_sources: Mapping[str, Any]) -> OrderedDict[str, Any]:
    tracks: OrderedDict[str, Any] = OrderedDict()
    for track_name, dataset in track_sources.items():
        if dataset is None or len(dataset) == 0:
            continue
        for bucket_name, subset in split_dataset_by_content(dataset).items():
            tracks[f"{track_name}.{bucket_name}"] = subset
    return tracks


def safe_delta(current: float | int | None, previous: float | int | None) -> float | None:
    if current is None or previous is None:
        return None
    if not isinstance(current, (int, float)) or not isinstance(previous, (int, float)):
        return None
    if not math.isfinite(float(current)) or not math.isfinite(float(previous)):
        return None
    return float(current) - float(previous)


def summarize_checkpoint_metrics(
    checkpoint_name: str,
    metrics_by_track: Mapping[str, Mapping[str, Any]],
    *,
    base_metrics: Mapping[str, Mapping[str, Any]] | None = None,
    stage1_metrics: Mapping[str, Mapping[str, Any]] | None = None,
    retention_track_prefix: str = "retention_shisa",
    eval_loss_tolerance: float = 0.20,
    perplexity_tolerance: float = 5.0,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "checkpoint": checkpoint_name,
        "tracks": {name: dict(values) for name, values in metrics_by_track.items()},
        "delta_vs_base": {},
        "delta_vs_stage1": {},
        "retention_regression_flags": [],
    }

    for track_name, metrics in metrics_by_track.items():
        if base_metrics and track_name in base_metrics:
            summary["delta_vs_base"][track_name] = {
                "eval_loss": safe_delta(metrics.get("eval_loss"), base_metrics[track_name].get("eval_loss")),
                "perplexity": safe_delta(metrics.get("perplexity"), base_metrics[track_name].get("perplexity")),
                "avg_tokens_per_sec": safe_delta(metrics.get("avg_tokens_per_sec"), base_metrics[track_name].get("avg_tokens_per_sec")),
            }
        if stage1_metrics and track_name in stage1_metrics:
            summary["delta_vs_stage1"][track_name] = {
                "eval_loss": safe_delta(metrics.get("eval_loss"), stage1_metrics[track_name].get("eval_loss")),
                "perplexity": safe_delta(metrics.get("perplexity"), stage1_metrics[track_name].get("perplexity")),
                "avg_tokens_per_sec": safe_delta(metrics.get("avg_tokens_per_sec"), stage1_metrics[track_name].get("avg_tokens_per_sec")),
            }

    if stage1_metrics:
        for track_name, metrics in metrics_by_track.items():
            if not track_name.startswith(retention_track_prefix):
                continue
            prev = stage1_metrics.get(track_name)
            if not prev:
                continue
            eval_loss_delta = safe_delta(metrics.get("eval_loss"), prev.get("eval_loss"))
            perplexity_delta = safe_delta(metrics.get("perplexity"), prev.get("perplexity"))
            if eval_loss_delta is not None and eval_loss_delta > eval_loss_tolerance:
                summary["retention_regression_flags"].append(
                    f"{track_name}: eval_loss worsened by {eval_loss_delta:.4f} vs stage1"
                )
            if perplexity_delta is not None and perplexity_delta > perplexity_tolerance:
                summary["retention_regression_flags"].append(
                    f"{track_name}: perplexity worsened by {perplexity_delta:.4f} vs stage1"
                )

    return summary


def print_checkpoint_summary(summary: Mapping[str, Any]) -> None:
    checkpoint = summary.get("checkpoint", "unknown")
    print(f"\nCheckpoint summary: {checkpoint}")
    for track_name, metrics in (summary.get("tracks", {}) or {}).items():
        print(f"- {track_name}")
        print(f"  eval_loss={metrics.get('eval_loss'):.4f}" if isinstance(metrics.get("eval_loss"), (int, float)) else "  eval_loss=nan")
        print(f"  perplexity={metrics.get('perplexity'):.4f}" if isinstance(metrics.get("perplexity"), (int, float)) else "  perplexity=nan")
        print(f"  avg_tokens_per_sec={metrics.get('avg_tokens_per_sec'):.2f}" if isinstance(metrics.get("avg_tokens_per_sec"), (int, float)) else "  avg_tokens_per_sec=nan")

    flags = summary.get("retention_regression_flags", []) or []
    if flags:
        print("Retention regression flags:")
        for flag in flags:
            print(f"- {flag}")
    else:
        print("Retention regression flags: none")


def export_jsonl_rows(rows: list[Mapping[str, Any]], destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    return destination
