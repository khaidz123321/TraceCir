"""Transition Compiler (P0 protocol Sec. 5): a frozen vision-language model
that turns (reference image, modification text) into a holistic target
description plus atomic ADD/REMOVE/PRESERVE/REPLACE transition operations.

Default backbone: Qwen2.5-VL-7B-Instruct, 4-bit quantized -- chosen for
feasibility on a single consumer/Kaggle-class GPU (see project discussion);
any vision-language model reachable through this same call signature can be
swapped in.

IMPORTANT (Sec. 5.3 of the protocol): before running full P0 retrieval,
manually audit ~100 CIRCO + ~100 CIRR compiler outputs and confirm >=~85%
are labeled Correct. This module only produces the outputs; the audit
itself is a human judgment step this code cannot perform.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

# The protocol's Sec. 5.2 prompt, verbatim. Kept for the record only: run with
# Qwen2.5-VL-7B it made the model return a bare JSON array of atoms with no
# "target" (the Sec. 5.2 text never states the {"target", "atoms"} wrapper
# that Sec. 5.1 defines), so 220/220 CIRCO-val outputs were unparseable, and
# it split "plants instead of plungers" into REMOVE + ADD rather than REPLACE.
COMPILER_PROMPT_PROTOCOL_VERBATIM = (
    "You are given a reference image and a modification instruction.\n"
    "Decompose the requested visual change into atomic transition operations.\n"
    "Use only: ADD, REMOVE, PRESERVE, REPLACE.\n"
    "For each atom output: operation, source_state, target_state.\n"
    "Rules:\n"
    "1. Use visually observable phrases only.\n"
    "2. Do not add unsupported information.\n"
    "3. REPLACE must contain both source_state and target_state.\n"
    "4. ADD has target_state only.\n"
    "5. REMOVE has source_state only.\n"
    "6. PRESERVE has source_state only.\n"
    "7. Split compound modifications into independent atoms.\n"
    "8. Return JSON only."
)

# Used prompt (v4): the verbatim prompt plus (a) the operation definitions from
# the protocol's operation table, (b) the output schema from Sec. 5.1, (c) rules
# 9-14 and (d) five worked examples. Additions only; rules 1-8 are unchanged.
# This is the "improve the prompt/schema" step Sec. 5.3 allows, and it must be
# reported as a deviation from the verbatim Sec. 5.2 prompt.
# History: rules 11-14 came from a 10-query smoke test (source/target swapped,
# ADD phrase in source_state, viewpoint/zoom dropped, invented objects). A v3
# that stacked rules 15-20 on top (target-after-edit, PRESERVE for "same", REMOVE
# for negation, ...) made the 7B model over-apply them (reversed polarity, dropped
# atoms) and was worse than v2 in a side-by-side read of the 100 audited queries,
# so v4 returns to rules 1-14 and teaches the same behaviours by example instead.
# The examples are invented, not taken from any CIRCO/CIRR query. The prompt is
# tuned against decomposition correctness (manual audit) only, never against
# retrieval metrics, and must be frozen before E0-E4 are run.
COMPILER_PROMPT = (
    "You are given a reference image and a modification instruction.\n"
    "Decompose the requested visual change into atomic transition operations.\n"
    "Use only: ADD, REMOVE, PRESERVE, REPLACE.\n"
    "For each atom output: operation, source_state, target_state.\n"
    "Operations:\n"
    "- ADD: something absent in the reference image that must appear in the target image.\n"
    "- REMOVE: something present in the reference image that must disappear.\n"
    "- PRESERVE: reference-image content that must remain.\n"
    "- REPLACE: a source visual state that must change into a different target state.\n"
    "Rules:\n"
    "1. Use visually observable phrases only.\n"
    "2. Do not add unsupported information.\n"
    "3. REPLACE must contain both source_state and target_state.\n"
    "4. ADD has target_state only.\n"
    "5. REMOVE has source_state only.\n"
    "6. PRESERVE has source_state only.\n"
    "7. Split compound modifications into independent atoms.\n"
    "8. Return JSON only.\n"
    "9. If something in the reference image is swapped for something else, output one REPLACE atom, "
    "not a separate REMOVE and ADD.\n"
    "10. Set every field that an operation does not use to null.\n"
    "11. The modification instruction describes the TARGET image. source_state is what the reference image "
    "shows; target_state is what the target image must show. Never swap them.\n"
    "12. For ADD, put the phrase in target_state and set source_state to null.\n"
    "13. Changes of viewpoint, zoom, camera angle, colour mode or style are also atoms: use REPLACE with "
    "the reference image's current value as source_state "
    '(for example REPLACE "shot from the side" -> "shot from above").\n'
    "14. Describe only what the reference image shows or the instruction states. Do not invent objects.\n"
    "Output format: one JSON object with exactly two keys.\n"
    '"target": one sentence describing the complete desired image after the change.\n'
    '"atoms": the list of atoms.\n'
    "Worked examples (the text in brackets stands for the reference image you would see):\n"
    "\n"
    "[Reference image: a red bicycle leaning on a brick wall]\n"
    "Instruction: is blue and has a basket\n"
    '{"target": "a blue bicycle with a basket leaning on a brick wall", "atoms": ['
    '{"operation": "REPLACE", "source_state": "red bicycle", "target_state": "blue bicycle"}, '
    '{"operation": "ADD", "source_state": null, "target_state": "basket"}]}\n'
    "\n"
    "[Reference image: a close-up of a white plate on a table]\n"
    "Instruction: has the same plate and is shot from a farther distance\n"
    '{"target": "a white plate on a table, shot from a farther distance", "atoms": ['
    '{"operation": "PRESERVE", "source_state": "white plate", "target_state": null}, '
    '{"operation": "REPLACE", "source_state": "close-up shot", "target_state": "shot from a farther distance"}]}\n'
    "\n"
    "[Reference image: a living room with a sofa, a lamp that is on, and a cat on the sofa]\n"
    "Instruction: has no cat and the lamp is off\n"
    '{"target": "a living room with a sofa and a lamp that is off", "atoms": ['
    '{"operation": "REMOVE", "source_state": "cat on the sofa", "target_state": null}, '
    '{"operation": "REPLACE", "source_state": "lamp on", "target_state": "lamp off"}]}\n'
    "\n"
    "[Reference image: a silver kettle on a stove, shot from the side]\n"
    "Instruction: is a different color and is seen from above\n"
    '{"target": "a kettle of a different colour on a stove, seen from above", "atoms": ['
    '{"operation": "REPLACE", "source_state": "silver kettle", "target_state": "kettle of a different colour"}, '
    '{"operation": "REPLACE", "source_state": "shot from the side", "target_state": "shot from above"}]}\n'
    "\n"
    "[Reference image: a lake with a wooden dock]\n"
    "Instruction: has two ducks on the water\n"
    '{"target": "a lake with a wooden dock and two ducks on the water", "atoms": ['
    '{"operation": "ADD", "source_state": null, "target_state": "two ducks on the water"}]}'
)

VALID_OPERATIONS = {"ADD", "REMOVE", "PRESERVE", "REPLACE"}


@dataclass
class TransitionAtom:
    operation: str
    source_state: str | None
    target_state: str | None


@dataclass
class TransitionSpec:
    target: str
    atoms: list[TransitionAtom] = field(default_factory=list)
    raw_output: str = ""
    parse_ok: bool = True

    def atoms_by_operation(self, operation: str) -> list[TransitionAtom]:
        return [a for a in self.atoms if a.operation == operation]

    def phrases(self, *operation_fields: tuple[str, str]) -> list[str]:
        """Non-empty phrases of the given (operation, field) pairs, e.g.
        `spec.phrases(("ADD", "target_state"), ("REPLACE", "target_state"))`."""
        out: list[str] = []
        for operation, field_name in operation_fields:
            for atom in self.atoms_by_operation(operation):
                value = getattr(atom, field_name)
                if value:
                    out.append(value)
        return out


def _clean_state(value) -> str | None:
    """Empty/blank strings and non-strings become None (models often write ""
    where the schema says null)."""
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return None


def load_compiler(
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    device: str = "cuda",
    load_in_4bit: bool = True,
):
    """Load the frozen vision-language compiler model + processor."""
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    kwargs = {"torch_dtype": torch.bfloat16, "device_map": device}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        kwargs.pop("device_map")
        kwargs["device_map"] = "auto"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **kwargs)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def _extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a model response, tolerating
    markdown code fences (```json ... ```) that instruction-tuned models
    commonly wrap structured output in despite "Return JSON only"."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    brace_match = re.search(r"\{.*\}", candidate, re.DOTALL)
    if not brace_match:
        return None
    try:
        return json.loads(brace_match.group(0))
    except json.JSONDecodeError:
        return None


def parse_transition_spec(raw_output: str) -> TransitionSpec:
    """Parse a compiler response into a TransitionSpec, per the Sec. 5.1
    schema. Malformed/missing fields fail soft (parse_ok=False, empty
    atoms) rather than raising, so a bad compiler call doesn't crash a
    full benchmark run -- but every such failure MUST surface in the
    manual audit (Sec. 5.3) and should count as "Incorrect" there.
    """
    payload = _extract_json(raw_output)
    target = _clean_state(payload.get("target")) if isinstance(payload, dict) else None
    if target is None:
        return TransitionSpec(target="", atoms=[], raw_output=raw_output, parse_ok=False)

    atoms: list[TransitionAtom] = []
    parse_ok = True
    raw_atoms = payload.get("atoms", [])
    if not isinstance(raw_atoms, list):
        raw_atoms, parse_ok = [], False
    for atom in raw_atoms:
        operation = atom.get("operation") if isinstance(atom, dict) else None
        operation = operation.strip().upper() if isinstance(operation, str) else operation
        if operation not in VALID_OPERATIONS:
            parse_ok = False
            continue
        atoms.append(
            TransitionAtom(
                operation=operation,
                source_state=_clean_state(atom.get("source_state")),
                target_state=_clean_state(atom.get("target_state")),
            )
        )

    return TransitionSpec(target=target, atoms=atoms, raw_output=raw_output, parse_ok=parse_ok)


def compile_query(
    model,
    processor,
    reference_image: Image.Image,
    modification: str,
    max_new_tokens: int = 512,
) -> TransitionSpec:
    """Run one (reference image, modification text) query through the
    compiler and return its parsed TransitionSpec.
    """
    import torch
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": reference_image},
                {"type": "text", "text": f"{COMPILER_PROMPT}\n\nModification instruction: {modification}"},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_trimmed = generated[:, inputs.input_ids.shape[1]:]
    raw_output = processor.batch_decode(
        generated_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return parse_transition_spec(raw_output)


def run_compiler_batch(
    model,
    processor,
    queries: list[tuple[Image.Image, str, str]],
    output_path: str | Path,
) -> list[dict]:
    """Run the compiler over a list of (reference_image, modification, query_id)
    tuples and cache the results (raw + parsed) to a JSONL file, for the
    manual audit step (Sec. 5.3) and for reuse across E0-E4.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    with open(output_path, "w", encoding="utf-8") as f:
        for reference_image, modification, query_id in queries:
            spec = compile_query(model, processor, reference_image, modification)
            record = {
                "query_id": query_id,
                "modification": modification,
                "target": spec.target,
                "atoms": [a.__dict__ for a in spec.atoms],
                "parse_ok": spec.parse_ok,
                "raw_output": spec.raw_output,
            }
            f.write(json.dumps(record) + "\n")
            records.append(record)
    return records


def load_compiled_queries(path: str | Path) -> dict[str, TransitionSpec]:
    """Load a cache produced by `run_compiler_batch`, keyed by query_id."""
    specs: dict[str, TransitionSpec] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            atoms = [TransitionAtom(**a) for a in record["atoms"]]
            specs[record["query_id"]] = TransitionSpec(
                target=record["target"],
                atoms=atoms,
                raw_output=record["raw_output"],
                parse_ok=record["parse_ok"],
            )
    return specs
