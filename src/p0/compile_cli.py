#!/usr/bin/env python
"""Headless Transition Compiler run (protocol Sec. 5) for a rented GPU box.

Resume-safe: every query's result is appended and flushed to the output
JSONL as soon as it is produced, and a rerun skips query_ids already in that
file. (`compiler.run_compiler_batch` rewrites the file from scratch on each
call, so an interrupted run there loses everything.)

The output format is identical to `run_compiler_batch`, so
`compiler.load_compiled_queries` / `run_e0` read it unchanged.

REMINDER (Sec. 5.3): the outputs must be manually audited (~100 CIRCO +
~100 CIRR, >= ~85% Correct) before trusting any retrieval numbers.

Example:
    nohup python -m src.p0.compile_cli \
        --dataset circo --data-root /workspace/data/CIRCO \
        --output-path /workspace/outputs/circo_compiled.jsonl \
        > /workspace/logs/compile.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Callable

from tqdm import tqdm

from ..data.datasets import CIRCODataset, CIRRDataset
from .compiler import _is_qwen3, compile_batch, compile_query, load_compiler
from .run_e0 import _query_id


def drop_torn_tail(output_path: str) -> None:
    """Remove a last line that lacks its trailing newline (a write cut short
    by a crash/kill). Left in place, the next appended record would be glued
    onto it, producing an unparseable line and silently losing that query."""
    if not os.path.exists(output_path):
        return
    with open(output_path, "rb+") as f:
        data = f.read()
        if not data or data.endswith(b"\n"):
            return
        f.truncate(data.rfind(b"\n") + 1)


def load_done_ids(output_path: str) -> set[str]:
    """query_ids already present in the output file (tolerates a torn last line)."""
    done: set[str] = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["query_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def compile_queries(
    items,
    compile_fn: Callable,
    output_path: str,
    dataset: str,
    limit: int | None = None,
    batch_size: int = 1,
    batch_compile_fn: Callable | None = None,
) -> int:
    """Compile every not-yet-done query and append its record to `output_path`.

    Args:
        items: iterable of dataset items, each with "reference_image" (PIL),
            "relative_caption" and the id fields `_query_id` needs.
        compile_fn: (image, modification) -> TransitionSpec.
        batch_size / batch_compile_fn: with batch_size > 1, up to `batch_size`
            pending queries are compiled together with
            batch_compile_fn(images, modifications) -> list[TransitionSpec];
            each record is still appended and flushed as soon as the batch ends.

    Returns:
        Number of queries newly compiled in this call.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    drop_torn_tail(output_path)
    done = load_done_ids(output_path)
    if done:
        print(f"Resume: {len(done)} cau da co trong {output_path}", flush=True)

    def pending():
        for index, item in enumerate(items):
            if limit is not None and index >= limit:
                return
            query_id = _query_id(dataset, item)
            if query_id not in done:
                yield query_id, item

    def write(out, query_id, item, spec) -> None:
        out.write(json.dumps({
            "query_id": query_id,
            "modification": item["relative_caption"],
            "target": spec.target,
            "atoms": [a.__dict__ for a in spec.atoms],
            "parse_ok": spec.parse_ok,
            "raw_output": spec.raw_output,
        }, ensure_ascii=False) + "\n")
        out.flush()

    newly = 0
    progress = tqdm(total=len(items) if limit is None else min(limit, len(items)), initial=len(done), desc="Compiling")
    with open(output_path, "a", encoding="utf-8") as out:
        batch: list = []
        for query_id, item in pending():
            if batch_size <= 1 or batch_compile_fn is None:
                write(out, query_id, item, compile_fn(item["reference_image"], item["relative_caption"]))
                newly += 1
                progress.update(1)
                continue
            batch.append((query_id, item))
            if len(batch) == batch_size:
                specs = batch_compile_fn([i["reference_image"] for _, i in batch], [i["relative_caption"] for _, i in batch])
                for (qid, it), spec in zip(batch, specs):
                    write(out, qid, it, spec)
                newly += len(batch)
                progress.update(len(batch))
                batch = []
        if batch:
            specs = batch_compile_fn([i["reference_image"] for _, i in batch], [i["relative_caption"] for _, i in batch])
            for (qid, it), spec in zip(batch, specs):
                write(out, qid, it, spec)
            newly += len(batch)
            progress.update(len(batch))
    progress.close()
    return newly


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["circo", "cirr"], required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Quantize to 4-bit (needed on ~12GB GPUs). Default: bf16, ~16GB VRAM.")
    parser.add_argument("--limit", type=int, default=None, help="Only compile the first N queries (smoke test).")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Queries per generate call (Qwen3.x only). Larger is faster; results can differ "
                             "slightly from batch size 1 because of padding.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Reference images must stay raw PIL for the VLM, so no CLIP preprocess.
    cls = CIRCODataset if args.dataset == "circo" else CIRRDataset
    items = cls(args.data_root, args.split, "relative", preprocess=lambda image: image)
    print(f"{len(items)} query ({args.dataset} {args.split})", flush=True)

    model, processor = load_compiler(args.model_name, device="cuda", load_in_4bit=args.load_in_4bit)
    print("Da nap compiler", args.model_name, flush=True)

    qwen3 = _is_qwen3(args.model_name)
    newly = compile_queries(
        items,
        lambda image, text: compile_query(model, processor, image, text, qwen3=qwen3),
        args.output_path,
        args.dataset,
        args.limit,
        batch_size=args.batch_size if qwen3 else 1,
        batch_compile_fn=(lambda images, texts: compile_batch(model, processor, images, texts)) if qwen3 else None,
    )

    done = load_done_ids(args.output_path)
    bad = 0
    with open(args.output_path, "r", encoding="utf-8") as f:
        for line in f:
            bad += 0 if json.loads(line)["parse_ok"] else 1
    print(f"COMPILE_DONE moi chay {newly} | tong {len(done)} | khong parse duoc {bad}", flush=True)


if __name__ == "__main__":
    main()
