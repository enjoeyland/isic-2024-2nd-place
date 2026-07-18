"""Compute GBDT OOF per-class accuracy (Benign/Malignant recall) and plot
like experiments/07_causal_discovery_comparison per-class charts.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path("/scratch2/[SC_LAB]/dataset/causal/skin/isic-2024-challenge")
GBDT_ROOT = Path(
    "/home/khmin1104/workspace/Medical-CausalInference/isic-2024-2nd-place/logs/gbdt/runs"
)
OUT_DIR = Path(
    "/home/khmin1104/workspace/Medical-CausalInference/isic-2024-2nd-place/logs/gbdt"
)
N_FOLD = 5
THRESHOLD = 0.5

# (display_name, run_dir_name)
RUNS = [
    ("0NNs-1types", "260709-0NNs-1types-feV7-s5-tuning_weights"),
    ("0NNs-18types", "260709-0NNs-18types-feV7-s5-tuning_weights"),
    ("1NNs-1types\nconvnext", "260714-1NNs-1types-convnext-feV7-s5-tuning_weights"),
    ("1NNs-1types\ntip_conv", "260714-1NNs-1types-tip_conv-feV7-s5-tuning_weights"),
    ("1NNs-18types", "260709-1NNs-18types-feV7-s5-tuning_weights"),
    ("2NNs-17types", "260714-2NNs-17types-feV7-s5-tuning_weights"),
    ("7NNs-18types", "260709-7NNs-18types-feV7-s5-tuning_weights"),
    ("9NNs-18types", "0906-9NNs-18types-feV7-s5-tuning_weights"),
]

THEME_PALETTE = ["#0072B2", "#009E73"]


def load_oof_predictions(run_dir: Path, df_fold: pd.DataFrame) -> pd.DataFrame:
    """True OOF: for each fold, keep only that fold's validation rows."""
    parts = []
    for k in range(N_FOLD):
        col = f"TSGKF_{N_FOLD}_{k}"
        val_ids = set(df_fold.loc[df_fold[col] == "val", "isic_id"])
        df_pred = pd.read_parquet(run_dir / f"fold{k}.parquet")
        parts.append(df_pred[df_pred["isic_id"].isin(val_ids)])
    oof = pd.concat(parts, ignore_index=True)
    # one prediction per id
    oof = oof.drop_duplicates(subset=["isic_id"], keep="first")
    return oof


def per_class_recall(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    classes = ["Benign", "Malignant"]
    class_counts = {
        "Benign": int((y_true == 0).sum()),
        "Malignant": int((y_true == 1).sum()),
    }
    per_class_acc = {}
    for c, label in enumerate(classes):
        mask = y_true == c
        per_class_acc[label] = float((y_pred[mask] == c).mean()) if mask.any() else float("nan")
    overall = float((y_pred == y_true).mean())
    return {
        "classes": classes,
        "class_counts": class_counts,
        "per_class_acc": per_class_acc,
        "overall_acc": overall,
    }


def plot_per_class(results: dict[str, dict], out_path: Path, title: str) -> None:
    method_names = list(results.keys())
    first = next(iter(results.values()))
    classes = first["classes"]
    class_counts = first["class_counts"]
    n_classes = len(classes)

    rows = [results[m]["per_class_acc"] for m in method_names]
    x = np.arange(len(method_names))
    width = 0.8 / n_classes
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(method_names)), 4.8))
    for i, c in enumerate(classes):
        vals = [row[c] for row in rows]
        offset = (i - (n_classes - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=f"{c} (n={class_counts[c]})",
            color=THEME_PALETTE[i],
            edgecolor="black",
            linewidth=0.6,
        )
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=20, ha="right")
    ax.set_ylabel("Per-class held-out accuracy (recall)")
    ax.set_ylim(0, 1.08)
    ax.set_title(title, fontsize=13)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=n_classes, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    meta = pd.read_csv(DATA_DIR / "train-metadata.csv", low_memory=False)[["isic_id", "target"]]
    df_fold = pd.read_parquet(DATA_DIR / "df_train_preprocessed.parquet")
    assert set(meta["isic_id"]) == set(df_fold["isic_id"])

    results: dict[str, dict] = {}
    table_rows = []

    for display_name, run_name in RUNS:
        run_dir = GBDT_ROOT / run_name
        if not (run_dir / "fold0.parquet").exists():
            print(f"skip missing: {run_name}")
            continue
        oof = load_oof_predictions(run_dir, df_fold)
        merged = oof.merge(meta, on="isic_id", how="inner")
        assert len(merged) == len(meta), (len(merged), len(meta))

        metrics = per_class_recall(
            merged["target"].to_numpy(),
            merged["predictions"].to_numpy(),
            threshold=THRESHOLD,
        )
        results[display_name] = metrics
        table_rows.append(
            {
                "config": display_name.replace("\n", " "),
                "run": run_name,
                "n": len(merged),
                "benign_recall": metrics["per_class_acc"]["Benign"],
                "malignant_recall": metrics["per_class_acc"]["Malignant"],
                "overall_acc": metrics["overall_acc"],
                "n_benign": metrics["class_counts"]["Benign"],
                "n_malignant": metrics["class_counts"]["Malignant"],
                "threshold": THRESHOLD,
            }
        )
        print(
            f"{display_name.replace(chr(10), ' '):30s}  "
            f"Benign={metrics['per_class_acc']['Benign']:.4f}  "
            f"Malignant={metrics['per_class_acc']['Malignant']:.4f}  "
            f"overall={metrics['overall_acc']:.4f}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_table = pd.DataFrame(table_rows)
    csv_path = OUT_DIR / "gbdt_per_class_accuracy.csv"
    df_table.to_csv(csv_path, index=False)

    png_path = OUT_DIR / "gbdt_per_class_accuracy.png"
    plot_per_class(
        results,
        png_path,
        f"ISIC 2024 GBDT: per-class OOF accuracy (threshold={THRESHOLD})",
    )
    print(f"\nwrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
