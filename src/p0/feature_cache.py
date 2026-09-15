"""Feature extraction + cache for P0 (protocol Sec. 4).

Builds and persists, for every image in a dataset's classic/index split:
    features/<dataset>/global.npy    # [N, d], float32
    features/<dataset>/local.npy     # [N, M, d], float16
    features/<dataset>/image_ids.json
    features/<dataset>/_progress.json   # {"completed": k} -- resume marker

One shared cache is built per dataset and reused across E0-E4 (Sec. 4:
"Use one shared cache for E0-E4 to guarantee a matched comparison.").

Memory note: for CIRCO's 123,403 images, the local-vector array alone is
123403 * 64 * 768 * 4 bytes =~ 22.6 GB in float32 -- comfortably larger
than a typical Colab/Kaggle instance's system RAM. This implementation
writes directly into disk-backed `np.memmap` arrays batch by batch (never
holding more than one batch in a Python list) and stores local vectors as
float16 (~11.3 GB total) to roughly halve that footprint; global vectors
stay float32 since they're small (~360 MB total) and are compared directly
against text features (Eq. 6), where precision matters a bit more.

It also checkpoints progress (`_progress.json`) after every batch, so a
killed/disconnected session (Colab/Kaggle free-tier idle timeouts, OOM
kills, etc.) can be resumed with the exact same call instead of restarting
from image 0.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from ..models.openclip_utils import encode_image_global_local


def _progress_path(output_dir: Path) -> Path:
    return output_dir / "_progress.json"


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
    feature_dim: int | None = None,
) -> None:
    """Extract global+local features for every item in `dataset` (must be
    a "classic" split dataset yielding {"image": tensor, id_key: value},
    in a fixed/deterministic order) and save them to `output_dir`.

    Safe to re-run after an interruption: it detects a partially-built
    cache via `_progress.json` and resumes from the first unprocessed
    image instead of starting over.

    Image identifiers are stored in the exact order they were processed
    (Sec. 4: "Store image identifiers in the exact evaluation order."),
    matching row order of global.npy / local.npy.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(dataset)
    if feature_dim is None:
        feature_dim = model.visual.output_dim if hasattr(model.visual, "output_dim") else 768
    m_local = local_grid * local_grid

    global_path = output_dir / "global.npy"
    local_path = output_dir / "local.npy"
    ids_path = output_dir / "image_ids.json"
    progress_path = _progress_path(output_dir)

    # `image_ids` are plain metadata (no model inference needed) -- write
    # them once up front so a resumed run doesn't need to recompute them.
    if not ids_path.exists():
        all_ids = getattr(dataset, "image_ids", None)
        if all_ids is None:
            all_ids = [dataset[i][id_key] for i in range(n_total)]
        with open(ids_path, "w", encoding="utf-8") as f:
            json.dump(list(all_ids), f)

    completed = 0
    if progress_path.exists() and global_path.exists() and local_path.exists():
        with open(progress_path, "r", encoding="utf-8") as f:
            completed = json.load(f)["completed"]
        print(f"Resuming feature cache from image {completed}/{n_total}")

    # `np.lib.format.open_memmap` (unlike a raw `np.memmap`) writes a
    # standard .npy file complete with header/dtype/shape metadata, so the
    # same file can later be opened with plain `np.load(..., mmap_mode="r")`
    # in FeatureCache -- a raw np.memmap file has no such header and isn't
    # np.load-compatible.
    global_arr = np.lib.format.open_memmap(
        global_path, mode="r+" if completed else "w+", dtype=np.float32, shape=(n_total, feature_dim)
    )
    local_arr = np.lib.format.open_memmap(
        local_path, mode="r+" if completed else "w+", dtype=np.float16, shape=(n_total, m_local, feature_dim)
    )

    if completed >= n_total:
        print("Feature cache already complete.")
        global_arr.flush()
        local_arr.flush()
        return

    remaining = Subset(dataset, list(range(completed, n_total)))
    loader = DataLoader(remaining, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    row = completed
    for batch in tqdm(loader, desc=f"Caching features -> {output_dir}", initial=completed // batch_size):
        images = batch["image"].to(device)
        v0, v_local = encode_image_global_local(model, images, local_grid=local_grid)

        n = v0.shape[0]
        global_arr[row : row + n] = v0.cpu().numpy().astype(np.float32)
        local_arr[row : row + n] = v_local.cpu().numpy().astype(np.float16)
        row += n

        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump({"completed": row}, f)

    global_arr.flush()
    local_arr.flush()
    print(f"Feature cache complete: {row}/{n_total} images -> {output_dir}")


class FeatureCache:
    """Read-only view over a cache built by `build_feature_cache`, with an
    id -> row-index lookup for O(1) access during scoring.

    Uses `mmap_mode="r"` so the (potentially multi-GB) local-vector array
    is not fully loaded into RAM -- rows are paged in from disk on demand.
    """

    def __init__(self, cache_dir: str | Path):
        cache_dir = Path(cache_dir)
        self.global_features = np.load(cache_dir / "global.npy", mmap_mode="r")
        self.local_features = np.load(cache_dir / "local.npy", mmap_mode="r")
        with open(cache_dir / "image_ids.json", "r", encoding="utf-8") as f:
            self.image_ids: list = json.load(f)
        self._id_to_row = {image_id: i for i, image_id in enumerate(self.image_ids)}

    def __len__(self) -> int:
        return len(self.image_ids)

    def global_vector(self, image_id) -> torch.Tensor:
        return torch.from_numpy(np.asarray(self.global_features[self._id_to_row[image_id]]))

    def local_vectors(self, image_id) -> torch.Tensor:
        return torch.from_numpy(np.asarray(self.local_features[self._id_to_row[image_id]])).float()

    def row_index(self, image_id) -> int:
        return self._id_to_row[image_id]
