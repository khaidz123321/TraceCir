# SEARLE baseline reproduction

Baseline re-implementation of **"Zero-Shot Composed Image Retrieval with
Textual Inversion"** (Baldrati, Agnolucci, Bertini, Del Bimbo — ICCV 2023,
[arXiv:2303.15247](https://arxiv.org/abs/2303.15247)), built from scratch
following the paper's equations and architecture, cross-checked against the
structure of the official repo ([miccunifi/SEARLE](https://github.com/miccunifi/SEARLE))
for directory/CLI conventions.

The paper's method (**SEARLE**) turns Composed Image Retrieval (an image +
a relative caption -> a target image) into plain text-to-image retrieval, by
mapping the reference image into a "pseudo-word" token that gets inserted
into the caption before it is encoded by CLIP's text tower.

## Repository layout

```
src/
  models/
    phi.py          Textual Inversion Network (Table 5): 3-layer MLP
    clip_utils.py   CLIP loading + encode_with_pseudo_tokens (token injection)
  data/
    datasets.py     FashionIQ / CIRR / CIRCO / unlabeled-image datasets
  losses.py         L_cos, L_gpt, L_distil (Eq. 1, 2, 4)
  concepts.py       CLIP zero-shot concept assignment + GPT-Neo phrase cache
  oti.py            Stage 1: Optimization-based Textual Inversion (Sec. 3.1)
  train_phi.py      Stage 2: distill OTI targets into Phi (Sec. 3.2)
  baselines.py      Training-free baselines: Image-only / Text-only / Image+Text
  eval.py           Recall@K, mAP@K (Eq. 6)
  scripts/
    run_oti.py      CLI: run stage 1 over a whole pre-training image set
    validate.py     CLI: stage 3, zero-shot CIR inference + evaluation
docs/
  DATASETS.md       Where to get FashionIQ / CIRR / CIRCO / ImageNet / vocab
```

## Pipeline

1. **Get data** — see [docs/DATASETS.md](docs/DATASETS.md). None of these
   datasets can be auto-downloaded (registration/agreement required).
2. **Stage 1 — OTI** (builds pseudo-word targets, run once, slow):
   ```bash
   python -m src.scripts.run_oti \
       --image-dir data/ImageNet1K/test \
       --vocab-path data/open_images_v7_classes.txt \
       --gpt-phrases-path data/gpt_phrases.jsonl \
       --output-path data/oti_targets.pt \
       --clip-model-name ViT-B/32
   ```
3. **Stage 2 — train Phi** (distillation, Sec. 3.2):
   ```bash
   python -m src.train_phi \
       --image-dir data/ImageNet1K/test \
       --oti-targets-path data/oti_targets.pt \
       --gpt-phrases-path data/gpt_phrases.jsonl \
       --concepts-path data/concepts.json \
       --output-dir checkpoints/phi_b32 \
       --clip-model-name ViT-B/32
   ```
4. **Stage 3 — zero-shot CIR evaluation**:
   ```bash
   python -m src.scripts.validate \
       --dataset cirr --split val --data-root data/CIRR \
       --phi-checkpoint checkpoints/phi_b32/phi_final.pt \
       --clip-model-name ViT-B/32
   ```

## Reference hyperparameters (Appendix A)

| Stage | Setting | Value |
|---|---|---|
| OTI | steps | 350 |
| OTI | learning rate | 2e-2 |
| OTI | lambda_cos / lambda_gpt | 1.0 / 0.5 |
| OTI | EMA decay | 0.99 |
| OTI | concepts per image (k) | 15 |
| Phi | epochs (B/32, L/14) | 100, 50 |
| Phi | learning rate | 1e-4 |
| Phi | batch size | 256 |
| Phi | lambda_distil / lambda_gpt | 1.0 / 0.75 |
| Phi | distillation temperature | 0.25 |
| Phi | concepts per image (k) | 150 |
| Both | optimizer | AdamW, weight decay 0.01 |

## Status / scope notes

This is a from-scratch **baseline reproduction**, not the official codebase.
Known simplifications, to be aware of before trusting exact numbers against
Tables 1-3 of the paper:

- `oti.py` batches images and shares one random template/GPT-phrase draw
  per optimization step across the whole batch loop, rather than the
  official repo's exact per-image scheduling; convergence behavior may
  differ slightly from the paper's reported numbers.
- CIRCO's mAP evaluation in `scripts/validate.py` needs each query's
  `gt_img_ids` mapped into index-set positions; wire this up against your
  local CIRCO annotation format before trusting mAP numbers (see the note
  printed by the script and `docs/DATASETS.md`).
- No test-set submission format for CIRCO's evaluation server is included;
  see https://circo.micc.unifi.it/ for that.
