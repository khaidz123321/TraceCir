"""Dataset classes for the two CIR benchmarks used in P0 (CIRR, CIRCO).

FashionIQ is explicitly out of scope for P0 (see docs/P0_PROTOCOL.md) and
there is no pre-training stage in the training-free TAPR/TRACE-CIR pipeline,
so no unlabeled-image / pre-training dataset classes are needed here.

Expected directory layouts (see docs/DATASETS.md for download instructions):

CIRR/
    cirr/captions/cap.rc2.{train,val,test1}.json
    cirr/image_splits/split.rc2.{train,val,test1}.json
    cirr/img_raw/{train,dev,test1}/...

CIRCO/
    annotations/{val,test}.json
    COCO2017_unlabeled/unlabeled2017/*.jpg
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset


def _pil_loader(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


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
