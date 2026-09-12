"""Concept assignment via CLIP zero-shot classification (Sec. 3.1) and
GPT-Neo phrase generation for the L_gpt regularization loss (Sec. 3.1 / A).

Pipeline:
  1. `assign_concepts`: for each image, embed it with CLIP and rank the
     ~20K Open Images V7 class names by cosine similarity; keep the top-k
     as that image's candidate "concepts".
  2. `pregenerate_gpt_phrases` (offline, run once): for every concept in the
     vocabulary, prompt GPT-Neo with "a photo of {concept}" and sample N
     continuations. These are cached to disk so OTI/Phi training can sample
     a phrase for a concept in O(1) instead of calling GPT at train time.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .models.clip_utils import tokenize


def load_open_images_vocab(vocab_path: str | Path) -> list[str]:
    """Load the ~20,932 Open Images V7 class names, one per line.

    See docs/DATASETS.md for how to export this list from the official
    Open Images V7 class-descriptions CSV.
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


@torch.no_grad()
def embed_vocab(clip_model, vocab: list[str], device, batch_size: int = 256) -> torch.Tensor:
    """Return L2-normalized CLIP text features for every vocabulary entry."""
    feats = []
    for i in tqdm(range(0, len(vocab), batch_size), desc="Embedding concept vocabulary"):
        batch = vocab[i : i + batch_size]
        prompts = [f"a photo of a {name}" for name in batch]
        tokens = tokenize(prompts).to(device)
        text_feats = clip_model.encode_text(tokens)
        feats.append(F.normalize(text_feats.float(), dim=-1))
    return torch.cat(feats, dim=0)


@torch.no_grad()
def assign_concepts(
    clip_model,
    image_loader: DataLoader,
    vocab: list[str],
    vocab_embeddings: torch.Tensor,
    device,
    top_k: int = 15,
) -> dict[str, list[str]]:
    """Assign the top-k most similar concept names to every image.

    Returns a mapping image_name -> [concept_1, ..., concept_k].
    """
    assignments: dict[str, list[str]] = {}
    for batch in tqdm(image_loader, desc="Assigning concepts"):
        images = batch["image"].to(device)
        names = batch["image_name"]
        image_feats = F.normalize(clip_model.encode_image(images).float(), dim=-1)
        sims = image_feats @ vocab_embeddings.t()  # (B, |vocab|)
        topk = sims.topk(top_k, dim=-1).indices.cpu()
        for name, idxs in zip(names, topk):
            assignments[name] = [vocab[i] for i in idxs.tolist()]
    return assignments


def pregenerate_gpt_phrases(
    vocab: list[str],
    output_path: str | Path,
    model_name: str = "EleutherAI/gpt-neo-2.7B",
    num_phrases_per_concept: int = 256,
    max_new_tokens: int = 35,
    temperature: float = 0.5,
    device: str = "cuda",
    batch_size: int = 32,
) -> None:
    """Pre-generate GPT-Neo continuations of "a photo of {concept}" for every
    concept in the vocabulary, and cache them to a JSON lines file.

    This mirrors Appendix A: run once (offline), ~12h on a single A100 for
    the full 20,932-concept Open Images vocabulary; the result is reused by
    both OTI and Phi training as a lookup table for L_gpt.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out_f:
        for concept in tqdm(vocab, desc="Pre-generating GPT phrases"):
            prompt = f"a photo of {concept}"
            phrases: list[str] = []
            for i in range(0, num_phrases_per_concept, batch_size):
                n = min(batch_size, num_phrases_per_concept - i)
                inputs = tokenizer([prompt] * n, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        do_sample=True,
                        temperature=temperature,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
                phrases.extend(decoded)
            out_f.write(json.dumps({"concept": concept, "phrases": phrases}) + "\n")


def load_gpt_phrases(path: str | Path) -> dict[str, list[str]]:
    """Load the JSONL cache produced by `pregenerate_gpt_phrases`."""
    phrases: dict[str, list[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            phrases[record["concept"]] = record["phrases"]
    return phrases
