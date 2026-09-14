"""Feature extraction + cache for P0 (protocol Sec. 4).

Builds and persists, for every image in a dataset's classic/index split:
    features/<dataset>/global.npy    # [N, d]
    features/<dataset>/local.npy     # [N, M, d]
    features/<dataset>/image_ids.json

One shared cache is built per dataset and reused across E0-E4 (Sec. 4:
"Use one shared cache for E0-E4 to guarantee a matched comparison.").
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.openclip_utils import encode_image_global_local


@torch.no_grad()
def build_feature_cache(
    model,
    dataset,
    output_dir: str | Path,
    id_key: str,
    device: str | torch.device,
    batch_size: int = 32,
    num_workers: int = 4,
    local_grid: int = 8,
) -> None:
    """Extract global+local features for every item in `dataset` (must be
    a "classic" split dataset yielding {"image": tensor, id_key: value})
    and save them to `output_dir`.

    Image identifiers are stored in the exact order they were processed
    (Sec. 4: "Store image identifiers in the exact evaluation order."),
    matching row order of global.npy / local.npy.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    all_global: list[np.ndarray] = []
    all_local: list[np.ndarray] = []
    all_ids: list = []

    for batch in tqdm(loader, desc=f"Caching features -> {output_dir}"):
        images = batch["image"].to(device)
        v0, v_local = encode_image_global_local(model, images, local_grid=local_grid)
        all_global.append(v0.cpu().numpy())
        all_local.append(v_local.cpu().numpy())
        ids = batch[id_key]
        all_ids.extend(ids.tolist() if hasattr(ids, "tolist") else ids)

    global_arr = np.concatenate(all_global, axis=0)
    local_arr = np.concatenate(all_local, axis=0)

    np.save(output_dir / "global.npy", global_arr)
    np.save(output_dir / "local.npy", local_arr)
    with open(output_dir / "image_ids.json", "w", encoding="utf-8") as f:
        json.dump(all_ids, f)


class FeatureCache:
    """Read-only view over a cache built by `build_feature_cache`, with an
    id -> row-index lookup for O(1) access during scoring."""

    def __init__(self, cache_dir: str | Path):
        cache_dir = Path(cache_dir)
        self.global_features = torch.from_numpy(np.load(cache_dir / "global.npy"))
        self.local_features = torch.from_numpy(np.load(cache_dir / "local.npy"))
        with open(cache_dir / "image_ids.json", "r", encoding="utf-8") as f:
            self.image_ids: list = json.load(f)
        self._id_to_row = {image_id: i for i, image_id in enumerate(self.image_ids)}

    def __len__(self) -> int:
        return len(self.image_ids)

    def global_vector(self, image_id) -> torch.Tensor:
        return self.global_features[self._id_to_row[image_id]]

    def local_vectors(self, image_id) -> torch.Tensor:
        return self.local_features[self._id_to_row[image_id]]

    def row_index(self, image_id) -> int:
        return self._id_to_row[image_id]
