#!/usr/bin/env python
"""Stage 2: train the Phi textual-inversion network by distilling the OTI
pseudo-word targets (Sec. 3.2, Eq. 4-5).

Example:
    python -m src.train_phi \
        --image-dir /data/ImageNet1K/test \
        --oti-targets-path data/oti_targets.pt \
        --gpt-phrases-path data/gpt_phrases.jsonl \
        --concepts-path data/concepts.json \
        --output-dir checkpoints/phi_b32 \
        --clip-model-name ViT-B/32
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .concepts import load_gpt_phrases
from .data.datasets import OTIDistillationDataset
from .losses import cosine_loss, distillation_contrastive_loss, phi_loss
from .models.clip_utils import build_oti_prompt, encode_with_pseudo_tokens, get_placeholder_token_id, load_clip
from .models.phi import Phi
from .seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--oti-targets-path", type=str, required=True)
    parser.add_argument("--gpt-phrases-path", type=str, required=True)
    parser.add_argument("--concepts-path", type=str, required=True,
                         help="JSON mapping image_name -> list[str] concepts (from concepts.assign_concepts).")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--clip-model-name", type=str, default="ViT-B/32")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--lambda-distil", type=float, default=1.0)
    parser.add_argument("--lambda-gpt", type=float, default=0.75)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    clip_model, preprocess = load_clip(args.clip_model_name, device)
    placeholder_id = get_placeholder_token_id(device)
    feature_dim = clip_model.token_embedding.embedding_dim

    dataset = OTIDistillationDataset(args.image_dir, args.oti_targets_path, preprocess)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=True,
    )

    with open(args.concepts_path, "r", encoding="utf-8") as f:
        concepts: dict[str, list[str]] = json.load(f)
    gpt_phrases = load_gpt_phrases(args.gpt_phrases_path)

    phi = Phi(input_dim=feature_dim, output_dim=feature_dim).to(device)
    optimizer = AdamW(phi.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.num_epochs):
        phi.train()
        running_loss = 0.0
        for batch in tqdm(loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}"):
            images = batch["image"].to(device)
            names = batch["image_name"]
            v_target = batch["oti_target"].to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                image_features = clip_model.encode_image(images).float()
            v_pred = phi(image_features)

            l_distil = distillation_contrastive_loss(v_pred, v_target, temperature=args.temperature)

            gpt_texts, gpt_star_texts = [], []
            for name in names:
                concept = random.choice(concepts[name])
                phrase_pool = gpt_phrases.get(concept, [f"a photo of {concept}"])
                phrase = random.choice(phrase_pool)
                gpt_texts.append(phrase)
                gpt_star_texts.append(phrase.replace(concept, "$", 1))

            from .models.clip_utils import tokenize

            gpt_tokens = tokenize(gpt_texts).to(device)
            with torch.no_grad():
                t_hat = F.normalize(clip_model.encode_text(gpt_tokens).float(), dim=-1)

            gpt_star_tokens = tokenize(gpt_star_texts).to(device)
            t_hat_star = encode_with_pseudo_tokens(clip_model, gpt_star_tokens, v_pred, placeholder_id)
            t_hat_star = F.normalize(t_hat_star, dim=-1)
            l_gpt = cosine_loss(t_hat, t_hat_star)

            loss = phi_loss(l_distil, l_gpt, args.lambda_distil, args.lambda_gpt)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        print(f"Epoch {epoch + 1}: avg loss = {avg_loss:.4f}")
        torch.save(phi.state_dict(), output_dir / f"phi_epoch{epoch + 1}.pt")

    torch.save(phi.state_dict(), output_dir / "phi_final.pt")


if __name__ == "__main__":
    main()
