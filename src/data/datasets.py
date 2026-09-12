"""Dataset classes for the three CIR benchmarks used in the paper
(FashionIQ, CIRR, CIRCO) plus a plain unlabeled-image dataset used to
pre-train OTI / the Phi network (e.g. ImageNet1K test split).

Expected directory layouts (see docs/DATASETS.md for download instructions):

FashionIQ/
    captions/cap.{dress,shirt,toptee}.{train,val,test}.json
    image_splits/split.{dress,shirt,toptee}.{train,val,test}.json
    images/*.jpg

CIRR/
    train/, dev/, test1/            (image folders)
    cirr/captions/cap.rc2.{train,val,test1}.json
    cirr/image_splits/split.rc2.{train,val,test1}.json

CIRCO/
    annotations/{val,test}.json
    COCO2017_unlabeled/unlabeled2017/*.jpg
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset


def _pil_loader(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def scan_image_dir(root: str | Path, extensions=(".jpg", ".jpeg", ".png")) -> dict[str, Path]:
    """Recursively scan `root` for images and return a deterministic
    filename-stem -> Path mapping, shared by every dataset that needs to
    resolve an "image name" to a file (`UnlabeledImageFolder`,
    `OTIDistillationDataset`).

    Using one shared, sorted implementation (rather than each dataset doing
    its own independent `rglob`) guarantees that two datasets built from the
    same directory agree on which physical file a given name resolves to.
    Raises if two files share a stem: silently letting one overwrite the
    other in the returned dict would let `run_oti.py` compute a pseudo-word
    target for one physical image while `OTIDistillationDataset` later feeds
    Phi a *different* image under the same name, corrupting distillation
    with no visible error.
    """
    root = Path(root)
    mapping: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in extensions:
            continue
        if path.stem in mapping:
            raise ValueError(
                f"Duplicate image stem '{path.stem}' found at both "
                f"{mapping[path.stem]} and {path}. Every image under {root} "
                "must have a unique filename (ignoring extension), since "
                "images are identified by stem throughout this pipeline "
                "(concept assignment, OTI targets, Phi distillation)."
            )
        mapping[path.stem] = path
    if not mapping:
        raise FileNotFoundError(f"No images found under {root}")
    return mapping


class UnlabeledImageFolder(Dataset):
    """Flat folder of images with no annotations, used to pre-train
    OTI / Phi (the paper uses the 100K images of ImageNet1K's test split)."""

    def __init__(self, root: str | Path, preprocess: Callable, extensions=(".jpg", ".jpeg", ".png")):
        self.root = Path(root)
        self.preprocess = preprocess
        name_to_path = scan_image_dir(self.root, extensions)
        self.names = sorted(name_to_path)
        self.paths = [name_to_path[name] for name in self.names]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        image = self.preprocess(_pil_loader(self.paths[idx]))
        return {"image": image, "image_name": self.names[idx]}


class OTIDistillationDataset(Dataset):
    """Pairs raw images with their pre-computed OTI pseudo-word targets
    (produced by `scripts/run_oti.py`), for distilling Phi (Sec. 3.2)."""

    def __init__(
        self,
        image_dir: str | Path,
        oti_targets_path: str | Path,
        preprocess: Callable,
    ):
        import torch

        self.image_dir = Path(image_dir)
        self.preprocess = preprocess

        cache = torch.load(oti_targets_path, map_location="cpu")
        self.image_names: list[str] = cache["image_names"]
        self.targets = cache["targets"]  # (N, d_w)

        self._name_to_path = scan_image_dir(self.image_dir)

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int):
        name = self.image_names[idx]
        image = self.preprocess(_pil_loader(self._name_to_path[name]))
        return {"image": image, "image_name": name, "oti_target": self.targets[idx]}


class FashionIQDataset(Dataset):
    """FashionIQ triplet / classic-split dataset.

    split: 'train' | 'val' | 'test'
    dress_types: subset of {'dress', 'shirt', 'toptee'}
    mode: 'relative' (ref image + relative caption[s]) or 'classic' (single
          image, used to build the retrieval index).
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        dress_types: list[str],
        mode: str,
        preprocess: Callable,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.mode = mode
        self.preprocess = preprocess

        self.triplets: list[dict] = []
        self.image_names: list[str] = []
        for dress_type in dress_types:
            cap_path = self.data_root / "captions" / f"cap.{dress_type}.{split}.json"
            with open(cap_path, "r", encoding="utf-8") as f:
                self.triplets.extend(json.load(f))

            split_path = self.data_root / "image_splits" / f"split.{dress_type}.{split}.json"
            with open(split_path, "r", encoding="utf-8") as f:
                self.image_names.extend(json.load(f))

    def _image_path(self, image_name: str) -> Path:
        return self.data_root / "images" / f"{image_name}.jpg"

    def __len__(self) -> int:
        return len(self.triplets) if self.mode == "relative" else len(self.image_names)

    def __getitem__(self, idx: int):
        if self.mode == "relative":
            triplet = self.triplets[idx]
            captions = triplet["captions"]  # list of 2 relative captions
            reference_name = triplet["candidate"]
            reference_image = self.preprocess(_pil_loader(self._image_path(reference_name)))
            item = {
                "reference_image": reference_image,
                "reference_name": reference_name,
                "relative_captions": captions,
            }
            if self.split != "test":
                target_name = triplet["target"]
                item["target_name"] = target_name
                item["target_image"] = self.preprocess(_pil_loader(self._image_path(target_name)))
            return item

        image_name = self.image_names[idx]
        image = self.preprocess(_pil_loader(self._image_path(image_name)))
        return {"image": image, "image_name": image_name}


class CIRRDataset(Dataset):
    """CIRR relative/classic dataset.

    split: 'train' | 'val' | 'test1'
    mode: 'relative' or 'classic'
    """

    def __init__(self, data_root: str | Path, split: str, mode: str, preprocess: Callable):
        self.data_root = Path(data_root)
        self.split = split
        self.mode = mode
        self.preprocess = preprocess

        cap_path = self.data_root / "cirr" / "captions" / f"cap.rc2.{split}.json"
        with open(cap_path, "r", encoding="utf-8") as f:
            self.triplets = json.load(f)

        split_path = self.data_root / "cirr" / "image_splits" / f"split.rc2.{split}.json"
        with open(split_path, "r", encoding="utf-8") as f:
            self.name_to_relpath: dict[str, str] = json.load(f)
        self.image_names = list(self.name_to_relpath.keys())

    def _image_path(self, image_name: str) -> Path:
        # image_splits/split.rc2.*.json maps a name to a path like
        # "./test1/xxx.png" or "./train/34/xxx.png", relative to the
        # official dataset's img_raw/ folder (see the CIRR repo's own
        # README "Dataset File Structure" section) -- not directly under
        # data_root/cirr/.
        relpath = self.name_to_relpath[image_name]
        if relpath.startswith("./"):
            relpath = relpath[2:]
        return self.data_root / "cirr" / "img_raw" / relpath

    def __len__(self) -> int:
        return len(self.triplets) if self.mode == "relative" else len(self.image_names)

    def __getitem__(self, idx: int):
        if self.mode == "relative":
            triplet = self.triplets[idx]
            reference_name = triplet["reference"]
            reference_image = self.preprocess(_pil_loader(self._image_path(reference_name)))
            item = {
                "reference_image": reference_image,
                "reference_name": reference_name,
                "relative_caption": triplet["caption"],
                "member_set": triplet["img_set"]["members"],
                "pair_id": triplet["pairid"],
            }
            if self.split != "test1":
                item["target_name"] = triplet["target_hard"]
            return item

        image_name = self.image_names[idx]
        image = self.preprocess(_pil_loader(self._image_path(image_name)))
        return {"image": image, "image_name": image_name}


class CIRCODataset(Dataset):
    """CIRCO relative/classic dataset (COCO 2017 unlabeled images).

    split: 'val' | 'test'
    mode: 'relative' or 'classic'
    """

    def __init__(self, data_root: str | Path, split: str, mode: str, preprocess: Callable):
        self.data_root = Path(data_root)
        self.split = split
        self.mode = mode
        self.preprocess = preprocess

        ann_path = self.data_root / "annotations" / f"{split}.json"
        with open(ann_path, "r", encoding="utf-8") as f:
            self.annotations = json.load(f)

        img_dir = self.data_root / "COCO2017_unlabeled" / "unlabeled2017"
        self.img_dir = img_dir
        self.image_ids = sorted(int(p.stem) for p in img_dir.glob("*.jpg"))

    def _image_path(self, image_id: int) -> Path:
        return self.img_dir / f"{image_id:012d}.jpg"

    def __len__(self) -> int:
        return len(self.annotations) if self.mode == "relative" else len(self.image_ids)

    def __getitem__(self, idx: int):
        if self.mode == "relative":
            ann = self.annotations[idx]
            reference_image = self.preprocess(_pil_loader(self._image_path(ann["reference_img_id"])))
            item = {
                "reference_image": reference_image,
                "reference_img_id": ann["reference_img_id"],
                "relative_caption": ann["relative_caption"],
                "shared_concept": ann.get("shared_concept", ""),
                "gt_img_ids": ann.get("gt_img_ids", []),
            }
            return item

        image_id = self.image_ids[idx]
        image = self.preprocess(_pil_loader(self._image_path(image_id)))
        return {"image": image, "image_id": image_id}
