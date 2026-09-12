"""Concept assignment via CLIP zero-shot classification (Sec. 3.1) and
GPT-Neo phrase generation for the L_gpt regularization loss (Sec. 3.1 / A).

Pipeline:
  1. `assign_concepts`: for each image, embed it with CLIP and rank the
     ~20K Open Images V7 class names by cosine similarity; keep the top-k
     as that image's candidate "concepts". `save_concepts`/`load_concepts`
     persist this mapping to/from the `concepts.json` file consumed by
     both `scripts/run_oti.py` (which produces it) and `train_phi.py`
     (which requires it).
  2. `pregenerate_gpt_phrases` (offline, run once): for every concept in the
     vocabulary, prompt GPT-Neo with "a photo of {concept}" and sample N
     continuations. These are cached to disk so OTI/Phi training can sample
     a phrase for a concept in O(1) instead of calling GPT at train time.
  3. `insert_pseudo_placeholder`: the single place that substitutes a
     concept word inside a GPT phrase with the "$" placeholder, shared by
     both OTI (oti.py) and Phi distillation (train_phi.py) so the two
     stages use identical substitution semantics (Sec. 3.1 / 3.2).
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .models.clip_utils import PSEUDO_TOKEN
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
            if name in assignments:
                raise ValueError(
                    f"Duplicate image name '{name}' encountered while assigning "
                    "concepts. Two different images resolved to the same name "
                    "(e.g. same filename stem in different subfolders), which "
                    "would silently overwrite one image's concepts with the "
                    "other's and later mismatch it against the wrong OTI "
                    "target. Rename the offending file(s) so every image has "
                    "a unique name."
                )
            assignments[name] = [vocab[i] for i in idxs.tolist()]
    return assignments


def save_concepts(concepts: dict[str, list[str]], output_path: str | Path) -> None:
    """Persist the image_name -> [concepts] mapping produced by
    `assign_concepts` to the `concepts.json` file `train_phi.py` expects."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(concepts, f)


def load_concepts(path: str | Path) -> dict[str, list[str]]:
    """Load the JSON cache produced by `save_concepts`."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def insert_pseudo_placeholder(phrase: str, concept: str, placeholder: str = PSEUDO_TOKEN) -> str | None:
    """Replace the first occurrence of `concept` in `phrase` with the
    pseudo-word placeholder, used to build the "$"-substituted counterpart
    of a GPT-generated phrase (Sec. 3.1, the T -> T* step in Fig. 2).

    The placeholder is always inserted surrounded by whitespace (then
    whitespace is collapsed) so that CLIP's BPE tokenizer always encodes it
    as the same isolated token used by `get_placeholder_token_id`, even
    when `concept` in the original phrase was immediately followed by
    punctuation (e.g. "a photo of dog, running..." -> "a photo of $ ,
    running..." rather than "a photo of $, running..." which BPE would
    tokenize differently).

    Returns None if `concept` cannot be found in `phrase` (e.g. GPT-Neo's
    decoding normalized whitespace/case differently than the seed prompt),
    so callers can skip that sample instead of building a phrase with zero
    placeholder occurrences and crashing later in `encode_with_pseudo_tokens`.
    """
    pattern = re.compile(re.escape(concept), re.IGNORECASE)
    if not pattern.search(phrase):
        return None
    replaced = pattern.sub(f" {placeholder} ", phrase, count=1)
    return re.sub(r"\s+", " ", replaced).strip()


def sample_gpt_pair(
    image_names: list[str],
    concepts: dict[str, list[str]],
    gpt_phrases: dict[str, list[str]],
    max_attempts: int = 5,
) -> tuple[list[str], list[str]]:
    """For each image, sample a concept + one of its pre-generated GPT
    phrases and build the "$"-substituted counterpart (Fig. 2's T -> T*).

    Shared by both OTI (`oti.py`) and Phi distillation (`train_phi.py`) so
    the two stages use identical substitution semantics, as the paper
    intends (Sec. 3.1 for OTI, Sec. 3.2 for Phi, both regularized by the
    same L_gpt).

    Retries with a different concept/phrase draw (up to `max_attempts`
    times) if `insert_pseudo_placeholder` can't find the concept verbatim
    in the sampled phrase, then falls back to the generic
    "a photo of {concept}" template (which always contains the concept
    verbatim by construction) rather than crashing.
    """
    gpt_texts: list[str] = []
    gpt_star_texts: list[str] = []
    for name in image_names:
        phrase, star = None, None
        for _ in range(max_attempts):
            concept = random.choice(concepts[name])
            phrase_pool = gpt_phrases.get(concept, [f"a photo of {concept}"])
            candidate = random.choice(phrase_pool)
            star = insert_pseudo_placeholder(candidate, concept)
            if star is not None:
                phrase = candidate
                break
        if star is None:
            concept = random.choice(concepts[name])
            phrase = f"a photo of {concept}"
            star = insert_pseudo_placeholder(phrase, concept)
        gpt_texts.append(phrase)
        gpt_star_texts.append(star)
    return gpt_texts, gpt_star_texts


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
