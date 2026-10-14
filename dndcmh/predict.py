"""Hash a single image or attribute vector."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dndcmh.models.adcmh import ADCMH
from dndcmh.models.necd import NECD


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--image", default=None, help="Path to face image")
    ap.add_argument("--attrs", default=None, help="Comma-separated 40 binary attrs (0/1)")
    ap.add_argument("--use-necd", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hash_bits = int(cfg["model"].get("hash_bits", 63))
    ckpt = torch.load(Path(cfg["paths"]["checkpoint_dir"]) / "last.pt", map_location=device, weights_only=False)

    model = ADCMH(hash_bits=hash_bits, pretrained=False).to(device)
    model.load_state_dict(ckpt["adcmh"])
    model.eval()
    necd = NECD(hash_bits=hash_bits).to(device)
    necd.load_state_dict(ckpt["necd"])
    necd.eval()

    out = {}
    with torch.no_grad():
        if args.image:
            tf = transforms.Compose(
                [
                    transforms.Resize((int(cfg["model"].get("image_size", 112)),) * 2),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
            x = tf(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
            z = model.encode_image(x)
            if args.use_necd:
                z = necd(z)
            code = torch.sign(z).cpu().numpy().astype(int).tolist()[0]
            out["image_hash"] = code
        if args.attrs:
            a = np.array([float(x) for x in args.attrs.split(",")], dtype=np.float32)
            if a.size != 40:
                raise SystemExit(f"expected 40 attrs, got {a.size}")
            z = model.encode_attr(torch.from_numpy(a).unsqueeze(0).to(device))
            if args.use_necd:
                z = necd(z)
            out["attr_hash"] = torch.sign(z).cpu().numpy().astype(int).tolist()[0]

    if not out:
        raise SystemExit("Provide --image and/or --attrs")

    text = json.dumps(out, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
