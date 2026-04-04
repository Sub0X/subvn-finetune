# Two-Stage VN Finetune

This repo trains a Gemma 4 LoRA adapter in two stages:

1. `Stage 1`: broad JA/EN instruction behavior from `shisa-ai/shisa-v2-sharegpt`
2. `Stage 2`: VN specialization from `lmg-anon/VNTL-v3.1-1k-q` with mandatory Shisa replay to reduce forgetting

The notebook now evaluates three checkpoints in order:

- `base`
- `stage1_shisa`
- `stage2_vntl_replay`

It also evaluates multiple tracks instead of a single held-out slice:

- `retention_shisa`: generic JA/EN retention holdout
- `vntl_q_dev`: representative `VNTL-v3.1-1k-q` dev sample
- `vntl_q_stress`: placeholder-heavy stress set from the tail of `-q`
- `vntl_harubench`: external non-`-q` VN track loaded from `assets/lmg-anon__VNTL-v3.1-1k.jsonl`

Each track is split into `general` and `explicit` subsets when non-empty. The notebook reports:

- per-track metrics
- `delta_vs_base`
- `delta_vs_stage1`
- `retention_regression_flags`

Helper logic for stratified track sampling and checkpoint summaries lives in [vn_finetune_utils.py](./vn_finetune_utils.py).

Post-generation VN evaluation helpers live in [vn_eval_suite.py](./vn_eval_suite.py). That script can score reference metrics plus targeted VN checks for placeholders, speaker tags, honorific alignment, inline markup preservation, and suspiciously short outputs.

The cleaned external VNTL eval corpus now lives in [assets/lmg-anon__VNTL-v3.1-1k.jsonl](./assets/lmg-anon__VNTL-v3.1-1k.jsonl), so the notebook no longer depends on an out-of-repo absolute path for that track.
