"""Grad-CAM visualization for a trained ConvNeXt(V2) ISIC lesion classifier.

The classification head of ``ISICModel`` (see
``src/models/components/timm_model_origin.py``) applies global pooling to the
last spatial feature map (``forward_features`` output, stored as
``model.features``) before the final linear layer. Since this feature map is
still spatial (B, C, H, W), a standard Grad-CAM can be computed directly from
it: backprop the target class logit to this feature map, average the
gradients spatially to get per-channel weights, and take the ReLU of the
weighted sum of channels.

Usage
-----
    conda activate isic_2nd
    cd isic-2024-2nd-place
    python scripts/generate_gradcam.py \
        --run_dir logs/train_all_data/runs/0821-convnextv2_tiny-meta_target03-transV2-lr1e-3-target_decay001-bs128_2-ep100-neg5-tsgkf \
        --n_pos 4 --n_neg 4

Outputs are written to ``<run_dir>/gradcam/`` by default: one overlay PNG per
sample plus a single ``gradcam_grid.png`` summary figure.
"""

from __future__ import annotations

import argparse
import os
from io import BytesIO

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.transforms import get_transforms  # noqa: E402
from src.models.components.timm_model_origin import ISICModel  # noqa: E402


def load_model(run_dir: str, ckpt_name: str, device: str):
    cfg = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))
    net_cfg = cfg.model.net

    model = ISICModel(
        model_name=net_cfg.model_name,
        num_classes=net_cfg.num_classes,
        pretrained=False,
        target_meta=bool(net_cfg.get("target_meta", False)),
    )

    ckpt_path = os.path.join(run_dir, "checkpoints", ckpt_name)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_state_dict = ckpt["state_dict"]

    state_dict = {}
    for key, value in raw_state_dict.items():
        if not key.startswith("net."):
            continue
        new_key = key[len("net.") :].replace("_orig_mod.", "")
        state_dict[new_key] = value

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] {len(missing)} missing keys, e.g. {missing[:5]}")
    if unexpected:
        print(f"[warn] {len(unexpected)} unexpected keys, e.g. {unexpected[:5]}")

    model.eval().to(device)
    return model, cfg


def pick_val_samples(cfg, n_pos: int, n_neg: int, seed: int) -> pd.DataFrame:
    data_dir = cfg.data.data_dir
    meta = pd.read_csv(os.path.join(data_dir, cfg.data.meta_csv_train_name))
    kfold_df = pd.read_parquet(os.path.join(data_dir, cfg.data.kfold_df_name))
    df = meta.merge(kfold_df, on="isic_id", how="left")

    kfold_method = cfg.data.kfold_method
    fold_col = {
        "sgkf": f"StratifiedGroupKFold_{cfg.data.n_fold}_{cfg.data.fold}",
        "gkf": f"GroupKFold_{cfg.data.n_fold}_{cfg.data.fold}",
        "tsgkf": f"TSGKF_{cfg.data.n_fold}_{cfg.data.fold}",
    }[kfold_method]

    val_df = df[df[fold_col] == "val"]
    rng = np.random.RandomState(seed)

    pos_pool = val_df[val_df["target"] == 1]
    neg_pool = val_df[val_df["target"] == 0]
    pos = pos_pool.sample(n=min(n_pos, len(pos_pool)), random_state=rng)
    neg = neg_pool.sample(n=min(n_neg, len(neg_pool)), random_state=rng)

    return pd.concat([pos, neg]).reset_index(drop=True)


def compute_gradcam(model: ISICModel, image_tensor: torch.Tensor, target_class: str, device: str):
    x = image_tensor.unsqueeze(0).to(device)
    logits = model(x)
    activations = model.features  # (1, C, H', W'), pre-pooling spatial features

    if target_class == "pred":
        cls = int(logits.argmax(dim=1).item())
    else:
        cls = int(target_class)

    score = logits[0, cls]
    grads = torch.autograd.grad(score, activations, retain_graph=False)[0]

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * activations).sum(dim=1, keepdim=True))
    cam = cam[0, 0].detach().cpu().numpy()

    cam -= cam.min()
    if cam.max() > 1e-8:
        cam /= cam.max()

    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    return cam, cls, probs


def overlay_cam(raw_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    h, w = raw_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap + (1 - alpha) * raw_rgb).astype(np.uint8)
    return overlay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True, help="e.g. logs/train_all_data/runs/<experiment_name>")
    parser.add_argument("--ckpt_name", default="last.ckpt")
    parser.add_argument("--n_pos", type=int, default=4, help="number of malignant (target=1) val samples")
    parser.add_argument("--n_neg", type=int, default=4, help="number of benign (target=0) val samples")
    parser.add_argument("--isic_ids", nargs="*", default=None, help="explicit isic_id list, overrides n_pos/n_neg")
    parser.add_argument(
        "--target_class",
        default="1",
        help="'pred' (use predicted class) or a class index (default '1' = malignant)",
    )
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.4, help="heatmap overlay opacity")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    out_dir = args.out_dir or os.path.join(run_dir, "gradcam")
    os.makedirs(out_dir, exist_ok=True)

    model, cfg = load_model(run_dir, args.ckpt_name, args.device)
    img_size = cfg.data.img_size
    _, transforms_test, transforms_type = get_transforms(cfg.data.transforms_version, img_size, finetuning=True)

    data_dir = cfg.data.data_dir
    if args.isic_ids:
        meta = pd.read_csv(os.path.join(data_dir, cfg.data.meta_csv_train_name))
        samples = meta[meta["isic_id"].isin(args.isic_ids)].reset_index(drop=True)
    else:
        samples = pick_val_samples(cfg, args.n_pos, args.n_neg, args.seed)

    if len(samples) == 0:
        raise RuntimeError("No samples found to visualize.")

    class_names = {0: "benign", 1: "malignant"}
    hdf5_path = os.path.join(data_dir, cfg.data.hdf5_train_name)

    ncols = 3
    nrows = len(samples)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if nrows == 1:
        axes = axes[None, :]

    with h5py.File(hdf5_path, "r") as fp_hdf:
        for i, row in samples.iterrows():
            isic_id = row["isic_id"]
            target = int(row["target"]) if "target" in row and not pd.isna(row["target"]) else -1

            raw = np.array(Image.open(BytesIO(fp_hdf[isic_id][()])).convert("RGB"))

            if transforms_type == "albumentations":
                tensor = transforms_test(image=raw)["image"]
            else:
                tensor = transforms_test(Image.fromarray(raw))

            cam, pred_cls, probs = compute_gradcam(model, tensor, args.target_class, args.device)

            raw_resized = cv2.resize(raw, (img_size, img_size))
            overlay = overlay_cam(raw_resized, cam, alpha=args.alpha)

            axes[i, 0].imshow(raw_resized)
            axes[i, 0].set_title(f"{isic_id}\ntarget={target} ({class_names.get(target, '?')})")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(cam, cmap="jet", vmin=0, vmax=1)
            axes[i, 1].set_title(f"Grad-CAM (class={pred_cls})")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(overlay)
            axes[i, 2].set_title(f"overlay | p(malignant)={probs[1]:.3f}")
            axes[i, 2].axis("off")

            cv2.imwrite(
                os.path.join(out_dir, f"{isic_id}_target{target}_gradcam.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            )

    plt.tight_layout()
    grid_path = os.path.join(out_dir, "gradcam_grid.png")
    plt.savefig(grid_path, dpi=150)
    plt.close(fig)

    print(f"Saved {len(samples)} individual overlays + grid to: {out_dir}")


if __name__ == "__main__":
    main()
