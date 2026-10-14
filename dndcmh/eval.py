"""Evaluate MAP / NDCG for single/double/triple attribute queries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dndcmh.data.celeba import CelebAAttrDataset, resolve_celeba_root
from dndcmh.metrics.retrieval import (
    build_attr_queries,
    mean_average_precision,
    ndcg_at_k,
)
from dndcmh.models.adcmh import ADCMH
from dndcmh.models.necd import NECD


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _device(cfg: dict, override: Optional[str] = None) -> torch.device:
    name = override or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


@torch.no_grad()
def extract_gallery(model: ADCMH, necd: Optional[NECD], loader, device, use_necd: bool):
    codes = []
    attrs = []
    for images, a, _ in tqdm(loader, desc="gallery", leave=False):
        images = images.to(device)
        z = model.encode_image(images)
        if use_necd and necd is not None:
            z = necd(z)
        codes.append(torch.sign(z).cpu().numpy())
        attrs.append(a.numpy())
    return np.concatenate(codes, 0), np.concatenate(attrs, 0)


@torch.no_grad()
def encode_queries(model: ADCMH, necd: Optional[NECD], q_attrs: np.ndarray, device, use_necd: bool):
    a = torch.from_numpy(q_attrs).to(device)
    z = model.encode_attr(a)
    if use_necd and necd is not None:
        z = necd(z)
    return torch.sign(z).cpu().numpy()


def evaluate(cfg: dict, device_override: Optional[str] = None) -> Dict:
    device = _device(cfg, device_override)
    root = resolve_celeba_root(cfg)
    seed = int(cfg["train"].get("seed", 42))
    max_n = cfg["data"].get("max_images")
    max_n = int(max_n) if max_n is not None else None
    image_size = int(cfg["model"].get("image_size", 112))
    hash_bits = int(cfg["model"].get("hash_bits", 63))

    ckpt_path = Path(cfg["paths"]["checkpoint_dir"]) / "last.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = ADCMH(
        hash_bits=hash_bits,
        attr_dim=int(cfg["model"].get("attr_dim", 40)),
        pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt["adcmh"])
    model.eval()
    necd = NECD(hash_bits=hash_bits).to(device)
    necd.load_state_dict(ckpt["necd"])
    necd.eval()

    # gallery = test split
    test_ds = CelebAAttrDataset(
        root, split="test", image_size=image_size, max_n=max_n, seed=seed
    )
    loader = DataLoader(
        test_ds,
        batch_size=int(cfg["eval"].get("batch_size", 128)),
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 4)),
    )

    n_q = int(cfg["eval"].get("n_queries_per_arity", 200))
    mq = cfg["data"].get("max_queries")
    if mq is not None:
        n_q = min(n_q, int(mq))
    top_k = int(cfg["eval"].get("top_k", 50))
    rng = np.random.RandomState(seed)

    results = {"n_gallery": len(test_ds), "hash_bits": hash_bits, "n_queries_per_arity": n_q}
    for use_necd, tag in [(False, "adcmh"), (True, "dndcmh")]:
        g_codes, g_attrs = extract_gallery(model, necd, loader, device, use_necd)
        tag_res = {}
        for arity, name in [(1, "single"), (2, "double"), (3, "triple")]:
            q_attrs, rel = build_attr_queries(g_attrs, arity, n_q, rng)
            q_codes = encode_queries(model, necd, q_attrs, device, use_necd)
            map_v = mean_average_precision(q_codes, g_codes, rel)
            ndcg_v = ndcg_at_k(q_codes, g_codes, rel, k=top_k)
            tag_res[name] = {"map": map_v, "ndcg": ndcg_v, "n_queries": int(q_attrs.shape[0])}
            print(f"[{tag}] {name}: MAP={map_v:.4f} NDCG@{top_k}={ndcg_v:.4f}")
        results[tag] = tag_res

    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / "eval_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(f"Wrote {out}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Eval DNDCMH MAP/NDCG")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    evaluate(load_config(args.config), device_override=args.device)


if __name__ == "__main__":
    main()
