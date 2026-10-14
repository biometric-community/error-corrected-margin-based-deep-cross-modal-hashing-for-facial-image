"""Build SVG figures + REPORT.md from real train/eval JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dndcmh.plot_style import (
    BAR_COLOR,
    FIGSIZE,
    LINEWIDTH_MAIN,
    MULTI_SERIES_COLORS,
    apply_rcparams,
    apply_style,
)


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def plot_train_loss(history, out: Path) -> None:
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    adcmh = [h for h in history if h.get("stage") == "adcmh"]
    necd = [h for h in history if h.get("stage") == "necd"]
    if adcmh:
        ax.plot(
            [h["epoch"] for h in adcmh],
            [h["total"] for h in adcmh],
            color=MULTI_SERIES_COLORS[0],
            linewidth=LINEWIDTH_MAIN,
            label="ADCMH",
        )
    if necd:
        ax.plot(
            [h["epoch"] for h in necd],
            [h["total"] for h in necd],
            color=MULTI_SERIES_COLORS[1],
            linewidth=LINEWIDTH_MAIN,
            label="NECD",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss")
    ax.legend()
    apply_style(ax)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def plot_map_bars(eval_res: dict, out: Path) -> None:
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    arities = ["single", "double", "triple"]
    x = np.arange(len(arities))
    w = 0.35
    ad = [eval_res.get("adcmh", {}).get(a, {}).get("map", 0) * 100 for a in arities]
    dn = [eval_res.get("dndcmh", {}).get(a, {}).get("map", 0) * 100 for a in arities]
    ax.bar(x - w / 2, ad, w, color=MULTI_SERIES_COLORS[0], label="ADCMH", edgecolor="#002080", linewidth=1)
    ax.bar(x + w / 2, dn, w, color=MULTI_SERIES_COLORS[1], label="DNDCMH", edgecolor="#800000", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(["Single", "Double", "Triple"])
    ax.set_ylabel("MAP (%)")
    ax.set_title("CelebA attribute→image MAP")
    ax.legend()
    apply_style(ax, grid_axis="y")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def write_report(cfg: dict, train_s: dict, eval_res: dict, fig_dir: Path, report_path: Path) -> None:
    protocol = cfg.get("train", {}).get("protocol", "full")
    lines = [
        f"# REPORT: DNDCMH — {protocol}",
        "",
        f"**Protocol:** `{protocol}`  ",
        f"**Package:** `dndcmh`  ",
        "",
        "## Summary",
        "",
        "| Method | Single MAP | Double MAP | Triple MAP |",
        "|--------|------------|------------|------------|",
    ]
    for tag, label in [("adcmh", "ADCMH"), ("dndcmh", "DNDCMH")]:
        row = eval_res.get(tag, {})
        s = row.get("single", {}).get("map", float("nan")) * 100
        d = row.get("double", {}).get("map", float("nan")) * 100
        t = row.get("triple", {}).get("map", float("nan")) * 100
        lines.append(f"| {label} | {s:.2f}% | {d:.2f}% | {t:.2f}% |")

    lines += [
        "",
        "## Training",
        "",
        f"- ADCMH epochs: {train_s.get('adcmh_epochs')}",
        f"- NECD epochs: {train_s.get('necd_epochs')}",
        f"- Train images: {train_s.get('n_train')}",
        f"- Hash bits: {train_s.get('hash_bits')}",
        f"- Margin m / ECC e: {train_s.get('margin_m')} / {train_s.get('ecc_e')}",
        f"- Elapsed: {train_s.get('elapsed_sec', 0):.2f} s",
        f"- Checkpoint: `{train_s.get('checkpoint')}`",
        "",
        "## Figures",
        "",
        "### `train_loss.svg`",
        "",
        f"![train_loss]({fig_dir.as_posix()}/train_loss.svg)",
        "",
        "### `map_by_arity.svg`",
        "",
        f"![map_by_arity]({fig_dir.as_posix()}/map_by_arity.svg)",
        "",
        "## Reproduce",
        "",
        "```bash",
        "bash scripts/smoke.sh   # or train_full.sh",
        "bash scripts/eval.sh configs/default.yaml",
        "bash scripts/report.sh",
        "```",
        "",
        "## Notes",
        "",
        "- Metrics from real CelebA train/eval only.",
        "- YTF not available; LFW attribute protocol deferred (see DEVIATIONS.md).",
        "",
    ]
    report_path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    log_dir = Path(cfg["paths"]["log_dir"])
    fig_dir = Path(cfg["paths"]["figure_dir"])
    train_s = load_json(log_dir / "train_summary.json") or {}
    eval_res = load_json(log_dir / "eval_results.json") or {}
    if not eval_res:
        raise SystemExit(f"Missing {log_dir / 'eval_results.json'}; run eval first")

    hist = train_s.get("history", [])
    plot_train_loss(hist, fig_dir / "train_loss.svg")
    plot_map_bars(eval_res, fig_dir / "map_by_arity.svg")

    # tracked copies for GitHub README embeds
    tracked = _ROOT / "figures"
    tracked.mkdir(exist_ok=True)
    for name in ("train_loss.svg", "map_by_arity.svg"):
        src = fig_dir / name
        if src.exists():
            (tracked / name).write_bytes(src.read_bytes())

    write_report(cfg, train_s, eval_res, Path("figures"), _ROOT / "REPORT.md")
    # also keep outputs path variant for local
    print(f"Wrote {_ROOT / 'REPORT.md'} and figures under {fig_dir}")


if __name__ == "__main__":
    main()
