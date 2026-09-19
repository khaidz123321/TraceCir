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

# E1 — explicit transition atoms + absolute matching (protocol Sec. 9.1), CIRCO val

Same cache, same compiled queries (Qwen3.6-27B, prompt v4), same 220 queries. E1: ADD = M(u+,I),
REMOVE = -M(u-,I), PRESERVE = M(u-,I), REPLACE = M(u+,I) - M(u-,I); S = Phi_T + 0.5 * mean(Phi_j);
M = tau_m log-sum-exp over the 64 local vectors, tau_m = 0.02. No reference-image term.

| | mAP@5 | mAP@10 | mAP@25 | mAP@50 |
|---|---|---|---|---|
| E0 (equal-weight placeholder) | 26.49 | 28.12 | 30.18 | 31.04 |
| E1, lambda_edit = 0.5 (default) | 24.51 | 25.61 | 27.79 | 28.64 |
| E1, lambda_edit = 0.25 | 23.75 | 25.06 | 27.29 | 28.03 |
| E1, lambda_edit = 1.0 | 24.44 | 25.06 | 27.50 | 28.37 |

Paired per-query AP, E1 - E0 (bootstrap, 10 000 resamples): mAP@5 -1.98, 95% CI [-4.78, +0.77];
mAP@10 -2.51, 95% CI [-5.11, +0.06]. **H1 is not supported** in this setting (E1 is not better than
E0; the difference is not significant either way).

mAP@5 by query group (groups overlap; from the 27B atoms): with PRESERVE (n = 96) E0 29.06 vs E1
23.85 (-5.21); REPLACE-only (n = 49) 31.55 vs 32.33 (+0.79); COMPOUND, >= 2 non-PRESERVE atoms
(n = 144) 25.47 vs 25.01 (-0.45); non-compound (n = 76) 28.42 vs 23.56 (-4.86).

Likely reason (a hypothesis, not tested): E0's Preserve uses the reference image's local features
(continuity), E1 by definition uses only the text of the source phrase, so E1 has no visual
evidence from the reference. That is what E2 (source grounding) adds back.
