# E0 — TAPR-style baseline, CIRCO validation (220 queries)

OpenCLIP ViT-L/14 (laion2b_s32b_b82k), 1 global + 8x8 local vectors, tau_local = 0.02,
factor weights theta_T = theta_A = theta_P = theta_R = 1 (placeholder: the TAPR paper does not
publish the numeric weights). Reference image removed from the ranking. mAP in %.

| Query side | mAP@5 | mAP@10 | mAP@25 | mAP@50 |
|---|---|---|---|---|
| No compiler (raw modification text as Target only) | 13.88 | 14.43 | 15.82 | 16.55 |
| Compiler: Qwen2.5-VL-7B, prompt v4 | 25.19 | 25.93 | 28.33 | 29.26 |
| **Compiler: Qwen3.6-27B, prompt v4 (candidate frozen)** | **26.49** | **28.12** | **30.18** | **31.04** |
| Reference: TAPR paper, ViT-L, CIRCO *test* | 37.83 | 38.46 | 41.44 | 42.50 |

Caveats: val (220 queries) is not the paper's test set; weights are a placeholder; the 27B vs 7B
gap (+1.3 mAP@5) is small for n = 220 and not tested for significance. E0 is the matched control
for E1-E4, not a test of H1-H3. The P0 protocol forbids large-scale hyperparameter tuning, so the
weights are not tuned here; ask the professor for the official values.

Command: `python -m src.p0.run_e0 --dataset circo --split val --data-root /workspace/data/CIRCO
--feature-cache-dir /workspace/features/circo --compiled-queries-path <compiled.jsonl>`
