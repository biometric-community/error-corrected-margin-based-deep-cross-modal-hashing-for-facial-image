"""CelebA attribute + image loader for cross-modal hashing."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

ATTR_NAMES: List[str] = [
    "5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young",
]


def resolve_celeba_root(cfg: dict) -> Path:
    raw = cfg.get("data", {}).get("celeba_root") or cfg.get("data", {}).get("root")
    if raw is None:
        raise FileNotFoundError("data.celeba_root missing in config")
    p = Path(raw)
    if not p.is_absolute():
        # configs are relative to project root
        proj = Path(__file__).resolve().parents[2]
        p = (proj / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"CelebA root not found: {p}")
    return p


def _find_images_dir(root: Path) -> Path:
    for cand in (
        root / "img_align_celeba",
        root / "Img" / "img_align_celeba",
        root,
    ):
        if cand.is_dir() and any(cand.glob("*.jpg")):
            return cand
    raise FileNotFoundError(f"No CelebA images under {root}")


class CelebAAttrDataset(Dataset):
    """Aligned CelebA faces with 40 binary attributes ({0,1})."""

    def __init__(
        self,
        root: Path,
        split: str = "train",
        image_size: int = 112,
        max_n: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        self.root = Path(root)
        self.img_dir = _find_images_dir(self.root)
        attr_path = self.root / "list_attr_celeba.txt"
        part_path = self.root / "list_eval_partition.txt"
        if not attr_path.exists():
            raise FileNotFoundError(attr_path)

        with open(attr_path) as f:
            _ = f.readline()  # count
            header = f.readline().strip().split()
            assert len(header) == 40, f"expected 40 attrs, got {len(header)}"
            rows = []
            for line in f:
                parts = line.split()
                name = parts[0]
                vals = np.array([1 if int(x) == 1 else 0 for x in parts[1:]], dtype=np.float32)
                rows.append((name, vals))

        part = {}
        if part_path.exists():
            with open(part_path) as f:
                for line in f:
                    name, p = line.split()
                    part[name] = int(p)

        # CelebA: 0=train, 1=val, 2=test
        want = {"train": 0, "val": 1, "test": 2}.get(split, 0)
        filtered = []
        for name, vals in rows:
            if part and part.get(name, 0) != want:
                continue
            filtered.append((name, vals))
        if not filtered and not part:
            # no partition file: random split 80/10/10
            rng = np.random.RandomState(seed)
            idx = rng.permutation(len(rows))
            n = len(rows)
            if split == "train":
                sel = idx[: int(0.8 * n)]
            elif split == "val":
                sel = idx[int(0.8 * n) : int(0.9 * n)]
            else:
                sel = idx[int(0.9 * n) :]
            filtered = [rows[i] for i in sel]

        if max_n is not None:
            filtered = filtered[: int(max_n)]

        self.names = [n for n, _ in filtered]
        self.attrs = np.stack([a for _, a in filtered], axis=0)
        self.tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        name = self.names[idx]
        img = Image.open(self.img_dir / name).convert("RGB")
        x = self.tf(img)
        a = torch.from_numpy(self.attrs[idx])
        return x, a, name
