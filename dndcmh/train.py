"""Train ADCMH then NECD on CelebA."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dndcmh.data.celeba import CelebAAttrDataset, resolve_celeba_root
from dndcmh.losses.dll import adcmh_loss
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


def train(cfg: dict, device_override: Optional[str] = None) -> Dict:
    device = _device(cfg, device_override)
    root = resolve_celeba_root(cfg)
    seed = int(cfg["train"].get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    max_n = cfg["data"].get("max_images")
    max_n = int(max_n) if max_n is not None else None
    image_size = int(cfg["model"].get("image_size", 112))
    hash_bits = int(cfg["model"].get("hash_bits", 63))
    margin_m = float(cfg["model"].get("margin_m", 6))
    ecc_e = int(cfg["model"].get("ecc_e", 6))

    train_ds = CelebAAttrDataset(
        root, split="train", image_size=image_size, max_n=max_n, seed=seed
    )
    loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"].get("batch_size", 64)),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    model = ADCMH(
        hash_bits=hash_bits,
        attr_dim=int(cfg["model"].get("attr_dim", 40)),
        pretrained=bool(cfg["model"].get("pretrained_backbone", True)),
    ).to(device)
    necd = NECD(hash_bits=hash_bits).to(device)

    lr = float(cfg["train"].get("lr", 1e-4))
    wd = float(cfg["train"].get("weight_decay", 1e-4))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    opt_necd = torch.optim.Adam(necd.parameters(), lr=lr, weight_decay=wd)

    theta = float(cfg["train"].get("theta", 1.0))
    lambda_reg = float(cfg["train"].get("lambda_reg", 0.1))
    gamma = float(cfg["train"].get("gamma", 1.0))
    adcmh_epochs = int(cfg["train"].get("adcmh_epochs", 20))
    necd_epochs = int(cfg["train"].get("necd_epochs", 10))

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    log_dir = Path(cfg["paths"]["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    history = []
    t0 = time.time()

    # ---- Stage 1: ADCMH ----
    model.train()
    for epoch in range(1, adcmh_epochs + 1):
        epoch_losses = []
        pbar = tqdm(loader, desc=f"ADCMH {epoch}/{adcmh_epochs}", leave=False)
        for images, attrs, _ in pbar:
            images = images.to(device, non_blocking=True)
            attrs = attrs.to(device, non_blocking=True)
            img_z, attr_z = model(images, attrs)
            loss, parts = adcmh_loss(
                img_z, attr_z, margin_m, theta=theta, lambda_reg=lambda_reg, gamma=gamma
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_losses.append(parts)
            pbar.set_postfix(loss=f"{parts['total']:.3f}")
        mean = {k: float(np.mean([d[k] for d in epoch_losses])) for k in epoch_losses[0]}
        mean["stage"] = "adcmh"
        mean["epoch"] = epoch
        history.append(mean)
        print(f"[ADCMH] epoch={epoch} loss={mean['total']:.4f} dll={mean['dll']:.4f}")

    # ---- Stage 2: NECD on ADCMH codes ----
    model.eval()
    necd.train()
    for epoch in range(1, necd_epochs + 1):
        epoch_losses = []
        pbar = tqdm(loader, desc=f"NECD {epoch}/{necd_epochs}", leave=False)
        for images, attrs, _ in pbar:
            images = images.to(device, non_blocking=True)
            attrs = attrs.to(device, non_blocking=True)
            with torch.no_grad():
                img_z, attr_z = model(images, attrs)
                # discrete ±1 targets (shared codeword proxy: average then sign)
                target = torch.sign(0.5 * (img_z + attr_z)).clamp(min=-1.0)
            noisy_img = NECD.flip_bits(target, ecc_e)
            noisy_attr = NECD.flip_bits(target, ecc_e)
            rec_img = necd(noisy_img)
            rec_attr = necd(noisy_attr)
            loss = F.mse_loss(rec_img, target) + F.mse_loss(rec_attr, target)
            # also pull decoded matched pair together
            loss = loss + F.mse_loss(necd(img_z.detach()), necd(attr_z.detach()))
            opt_necd.zero_grad(set_to_none=True)
            loss.backward()
            opt_necd.step()
            loss_v = float(loss.detach())
            epoch_losses.append(loss_v)
            pbar.set_postfix(loss=f"{loss_v:.3f}")
        mean_l = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        history.append({"stage": "necd", "epoch": epoch, "total": mean_l})
        print(f"[NECD] epoch={epoch} loss={mean_l:.4f}")

    elapsed = time.time() - t0
    ckpt = {
        "adcmh": model.state_dict(),
        "necd": necd.state_dict(),
        "config": cfg,
        "hash_bits": hash_bits,
        "margin_m": margin_m,
        "ecc_e": ecc_e,
    }
    torch.save(ckpt, ckpt_dir / "last.pt")
    summary = {
        "protocol": cfg["train"].get("protocol", "full"),
        "adcmh_epochs": adcmh_epochs,
        "necd_epochs": necd_epochs,
        "n_train": len(train_ds),
        "hash_bits": hash_bits,
        "margin_m": margin_m,
        "ecc_e": ecc_e,
        "elapsed_sec": elapsed,
        "history": history,
        "checkpoint": str(ckpt_dir / "last.pt"),
    }
    with open(log_dir / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"Wrote {log_dir / 'train_summary.json'} elapsed={elapsed:.1f}s")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Train DNDCMH (ADCMH + NECD)")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    train(cfg, device_override=args.device)


if __name__ == "__main__":
    main()
