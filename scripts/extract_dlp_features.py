"""Extract Deep-Learning-Parameter (DLP) node features from a trained
ConvNeXtV2 ISIC lesion classifier, for use as extra nodes in the medical
Bayesian network (experiments/06_bn_causal_explanation, 07_causal_discovery_
comparison).

Background
----------
Erlangga & Cho (2025) "Causally explainable AI ... for energy demand
prediction" add *deep learning parameters* (DLPs) -- class activation maps and
attention weights -- as extra nodes of a Bayesian network so the BN can
causally relate what the network internally computes to the prediction. Their
DLPs are (a) clustered CAMs ("CAM type 1/2") and (b) clustered attention
vectors. This script produces the medical-imaging analogue of those DLPs from
this project's ConvNeXtV2 classifier.

What this model gives us (and why it is unusually good for this)
----------------------------------------------------------------
The checkpoint was trained with ``target_meta=True``: besides the main
malignant/benign head, the network has auxiliary heads that reconstruct the
tabular metadata (lesion area, diameter, asymmetry, border, colour,
eccentricity, location, ...) *from the image alone*. That means the network was
explicitly pushed to encode the same ABCDE-style concepts the tabular BN uses
as nodes -- so its internal "DL-predicted asymmetry", "DL-predicted area", etc.
are image-derived counterparts of the exact BN feature nodes. This is what
makes a rigorous DL<->BN alignment (causal abstraction) tractable here, not
just an association-level DLP add-on.

DLP families produced (one row per lesion, keyed by isic_id)
------------------------------------------------------------
1. Prediction DLPs
   - ``dl_p_malignant``  : softmax P(malignant) from the main head
   - ``dl_logit_malignant``: raw margin logit_1 - logit_0
2. Embedding-cluster DLP (semantic "what")
   - ``dl_emb_cluster``  : K-means cluster id over the 768-d pooled penultimate
     embedding -- the direct analogue of the paper's K-means-over-DLP step
     (a learned "visual phenotype" of the lesion)
3. Activation-map DLPs (spatial "where"), from the pre-pool feature map
   - ``dl_cam_concentration``: how peaked the spatial activation is (max/mean of
     the channel-L2 activation-energy map) -- high = focused on one spot
   - ``dl_cam_peripherality`` : centre-vs-edge mass of that map -- high = the
     network reacts to lesion border/surround rather than its centre
   - ``dl_cam_cluster``       : K-means cluster id over the flattened map
     (paper's "CAM type" node)
4. Meta-head DLPs (image-derived counterparts of the BN's tabular nodes)
   - ``dlmeta_num_<col>``     : the network's image-only prediction of each of
     the 34 numeric metadata columns (standardised z-score space, as trained)
   - ``dlmeta_cat_<col>``     : argmax class id of each categorical meta head

Also written: ``embeddings.npy`` (N x 768) + ``embedding_isic_ids.npy`` for
downstream linear-probe / Distributed-Alignment-Search work (see
docs/260721-research_plan_dl_as_bn.md).

Usage
-----
    conda activate isic_2nd
    cd isic-2024-2nd-place
    # quick smoke test (few images):
    python scripts/extract_dlp_features.py --run_dir <run_dir> --limit 40
    # full BN subsample (all malignant + 4000 benign, matches datasets.py):
    python scripts/extract_dlp_features.py --run_dir <run_dir> --match_bn_subsample
"""

from __future__ import annotations

import argparse
import os
from io import BytesIO

import h5py
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from PIL import Image
from sklearn.cluster import KMeans

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.transforms import get_transforms  # noqa: E402
from src.models.components.timm_model_origin import ISICModel  # noqa: E402

# Numeric metadata columns for metadata_version=1, in the exact order the
# auxiliary head_meta_num was trained to output them (src/isic_utils/utils.py,
# prepare_df_for_dnn version 1). Index i of the head == NUM_COLS_V1[i].
NUM_COLS_V1 = [
    "age_approx", "clin_size_long_diam_mm", "tbp_lv_A", "tbp_lv_Aext", "tbp_lv_B",
    "tbp_lv_Bext", "tbp_lv_C", "tbp_lv_Cext", "tbp_lv_H", "tbp_lv_Hext", "tbp_lv_L",
    "tbp_lv_Lext", "tbp_lv_areaMM2", "tbp_lv_area_perim_ratio", "tbp_lv_color_std_mean",
    "tbp_lv_deltaA", "tbp_lv_deltaB", "tbp_lv_deltaL", "tbp_lv_deltaLB", "tbp_lv_deltaLBnorm",
    "tbp_lv_eccentricity", "tbp_lv_minorAxisMM", "tbp_lv_nevi_confidence", "tbp_lv_norm_border",
    "tbp_lv_norm_color", "tbp_lv_perimeterMM", "tbp_lv_radial_color_std_max", "tbp_lv_stdL",
    "tbp_lv_stdLExt", "tbp_lv_symm_2axis", "tbp_lv_symm_2axis_angle", "tbp_lv_x", "tbp_lv_y",
    "tbp_lv_z",
]
# Categorical meta heads for metadata_version=1, in head order (dims [3,2,21,8]).
CAT_COLS_V1 = ["sex", "tbp_tile_type", "tbp_lv_location", "tbp_lv_location_simple"]

# The subset of numeric meta heads that line up 1:1 with the tabular BN's
# morphology nodes (07_causal_discovery_comparison/datasets.py
# ISIC2024_MORPHOLOGY_CONT) -- these are the natural DL<->BN alignment targets.
BN_ALIGNED_NUM_COLS = [
    "clin_size_long_diam_mm", "tbp_lv_symm_2axis", "tbp_lv_norm_border",
    "tbp_lv_norm_color", "tbp_lv_color_std_mean", "tbp_lv_eccentricity", "tbp_lv_areaMM2",
]

ISIC2024_RANDOM_STATE = 42  # matches datasets.load_isic2024 so isic_ids join 1:1


def load_model(run_dir: str, ckpt_name: str, device: str):
    cfg = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))
    net_cfg = cfg.model.net
    model = ISICModel(
        model_name=net_cfg.model_name,
        num_classes=net_cfg.num_classes,
        pretrained=False,
        target_meta=bool(net_cfg.get("target_meta", False)),
    )
    ckpt = torch.load(os.path.join(run_dir, "checkpoints", ckpt_name), map_location="cpu", weights_only=False)
    state_dict = {}
    for key, value in ckpt["state_dict"].items():
        if not key.startswith("net."):
            continue
        state_dict[key[len("net.") :].replace("_orig_mod.", "")] = value
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] {len(missing)} missing keys, e.g. {missing[:5]}")
    if unexpected:
        print(f"[warn] {len(unexpected)} unexpected keys, e.g. {unexpected[:5]}")
    model.eval().to(device)
    return model, cfg


def pick_rows(cfg, args) -> pd.DataFrame:
    data_dir = cfg.data.data_dir
    meta = pd.read_csv(os.path.join(data_dir, cfg.data.meta_csv_train_name), low_memory=False)

    if args.match_bn_subsample:
        # Reproduce datasets.load_isic2024's stratified subsample so every
        # extracted isic_id joins the BN symbol table exactly.
        malignant = meta[meta["target"] == 1]
        benign = meta[meta["target"] == 0].sample(n=args.n_benign, random_state=ISIC2024_RANDOM_STATE)
        rows = pd.concat([malignant, benign], ignore_index=True)
    else:
        kfold_df = pd.read_parquet(os.path.join(data_dir, cfg.data.kfold_df_name))
        merged = meta.merge(kfold_df, on="isic_id", how="left")
        fold_col = {
            "sgkf": f"StratifiedGroupKFold_{cfg.data.n_fold}_{cfg.data.fold}",
            "gkf": f"GroupKFold_{cfg.data.n_fold}_{cfg.data.fold}",
            "tsgkf": f"TSGKF_{cfg.data.n_fold}_{cfg.data.fold}",
        }[cfg.data.kfold_method]
        rows = merged[merged[fold_col] == "val"]

    if args.limit:
        # keep class balance visible in a smoke test: take some of each class
        pos = rows[rows["target"] == 1].head(args.limit // 2)
        neg = rows[rows["target"] == 0].head(args.limit - len(pos))
        rows = pd.concat([pos, neg])
    return rows.reset_index(drop=True)


@torch.no_grad()
def forward_batch(model, images: torch.Tensor):
    """One forward pass yields everything (no backprop): main logits, pooled
    embedding, the pre-pool spatial feature map, and every meta-head output."""
    logits = model(images)                    # sets model.features (spatial, B x C x H x W)
    spatial = model.features
    pooled = model.pool(spatial)              # B x C  (768 for convnextv2_tiny)
    logits_num, logits_cat_list = model.forward_meta()
    return logits, pooled, spatial, logits_num, logits_cat_list


def activation_map_stats(spatial: torch.Tensor):
    """Class-agnostic spatial DLP stats from the pre-pool feature map, no
    backprop needed: per-cell activation energy = L2 norm across channels.
    Returns (concentration, peripherality, flattened_map) per sample.

    - concentration = max / mean of the energy map: peaked (one hot spot) vs flat.
    - peripherality = mean energy in the outer ring / mean energy in the centre
      block: >1 means the network responds more to the lesion surround/border
      than its centre. Grad-CAM overlays (scripts/generate_gradcam.py) are the
      qualitative companion to these scalar summaries.
    """
    b, c, h, w = spatial.shape
    energy = spatial.pow(2).sum(dim=1).sqrt()          # B x H x W
    flat = energy.flatten(1)                            # B x (H*W)
    mean = flat.mean(dim=1)
    mx = flat.max(dim=1).values
    concentration = (mx / (mean + 1e-6)).cpu().numpy()

    ch, cw = h // 3, w // 3
    centre = energy[:, ch : h - ch, cw : w - cw].reshape(b, -1).mean(dim=1)
    total = energy.reshape(b, -1).mean(dim=1)
    # outer mass = total - centre (area-weighted); ratio to centre mean
    peripherality = ((total * h * w - centre * (h - 2 * ch) * (w - 2 * cw))
                     / (centre * (h - 2 * ch) * (w - 2 * cw) + 1e-6)).cpu().numpy()

    return concentration, peripherality, flat.cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--ckpt_name", default="last.ckpt")
    parser.add_argument("--out_dir", default=None, help="default: <run_dir>/dlp_features")
    parser.add_argument("--match_bn_subsample", action="store_true",
                        help="reproduce datasets.load_isic2024 sampling (all malignant + n_benign) so rows join the BN")
    parser.add_argument("--n_benign", type=int, default=4000)
    parser.add_argument("--limit", type=int, default=0, help="cap #rows (smoke test); 0 = no cap")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--emb_clusters", type=int, default=6, help="K for embedding K-means DLP")
    parser.add_argument("--cam_clusters", type=int, default=2, help="K for activation-map K-means DLP (paper used 2)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    out_dir = args.out_dir or os.path.join(run_dir, "dlp_features")
    os.makedirs(out_dir, exist_ok=True)

    model, cfg = load_model(run_dir, args.ckpt_name, args.device)
    img_size = cfg.data.img_size
    _, transforms_test, transforms_type = get_transforms(cfg.data.transforms_version, img_size, finetuning=True)

    rows = pick_rows(cfg, args)
    print(f"extracting DLP features for {len(rows)} lesions "
          f"(malignant={int(rows['target'].sum())}, device={args.device})")

    data_dir = cfg.data.data_dir
    hdf5_path = os.path.join(data_dir, cfg.data.hdf5_train_name)

    records: list[dict] = []
    embeddings: list[np.ndarray] = []
    cam_flats: list[np.ndarray] = []

    with h5py.File(hdf5_path, "r") as fp_hdf:
        batch_imgs, batch_meta = [], []

        def flush():
            if not batch_imgs:
                return
            images = torch.stack(batch_imgs).to(args.device)
            logits, pooled, spatial, logits_num, logits_cat_list = forward_batch(model, images)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            margin = (logits[:, 1] - logits[:, 0]).cpu().numpy()
            conc, periph, cam_flat = activation_map_stats(spatial)
            num_pred = logits_num.cpu().numpy()
            cat_pred = [c.argmax(dim=1).cpu().numpy() for c in logits_cat_list]

            for i, meta_row in enumerate(batch_meta):
                rec = {
                    "isic_id": meta_row["isic_id"],
                    "target": int(meta_row["target"]),
                    "dl_p_malignant": float(probs[i]),
                    "dl_logit_malignant": float(margin[i]),
                    "dl_cam_concentration": float(conc[i]),
                    "dl_cam_peripherality": float(periph[i]),
                }
                for j, col in enumerate(NUM_COLS_V1):
                    rec[f"dlmeta_num_{col}"] = float(num_pred[i, j])
                for k, col in enumerate(CAT_COLS_V1):
                    rec[f"dlmeta_cat_{col}"] = f"c{int(cat_pred[k][i])}"
                records.append(rec)
                embeddings.append(pooled[i].cpu().numpy())
                cam_flats.append(cam_flat[i])
            batch_imgs.clear()
            batch_meta.clear()

        for _, row in rows.iterrows():
            raw = np.array(Image.open(BytesIO(fp_hdf[row["isic_id"]][()])).convert("RGB"))
            if transforms_type == "albumentations":
                tensor = transforms_test(image=raw)["image"]
            else:
                tensor = transforms_test(Image.fromarray(raw))
            batch_imgs.append(tensor)
            batch_meta.append(row)
            if len(batch_imgs) >= args.batch_size:
                flush()
        flush()

    df = pd.DataFrame(records)
    emb = np.stack(embeddings)
    cam = np.stack(cam_flats)

    # Cluster DLPs (paper's K-means-over-DLP step). Guard against asking for
    # more clusters than samples in a small smoke run.
    k_emb = min(args.emb_clusters, len(df))
    k_cam = min(args.cam_clusters, len(df))
    if k_emb >= 2:
        df["dl_emb_cluster"] = KMeans(n_clusters=k_emb, random_state=42, n_init=10).fit_predict(emb).astype(str)
    else:
        df["dl_emb_cluster"] = "0"
    if k_cam >= 2:
        df["dl_cam_cluster"] = KMeans(n_clusters=k_cam, random_state=42, n_init=10).fit_predict(cam).astype(str)
    else:
        df["dl_cam_cluster"] = "0"

    df.to_parquet(os.path.join(out_dir, "dlp_features.parquet"), index=False)
    df.to_csv(os.path.join(out_dir, "dlp_features.csv"), index=False)
    np.save(os.path.join(out_dir, "embeddings.npy"), emb)
    np.save(os.path.join(out_dir, "embedding_isic_ids.npy"), df["isic_id"].to_numpy())

    print(f"wrote {len(df)} rows x {df.shape[1]} cols to {out_dir}/dlp_features.parquet")
    print(f"      embeddings.npy shape = {emb.shape}")
    print("BN-aligned meta-head columns available:",
          [f"dlmeta_num_{c}" for c in BN_ALIGNED_NUM_COLS])
    print(df[["isic_id", "target", "dl_p_malignant", "dl_emb_cluster",
              "dl_cam_cluster", "dl_cam_concentration"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
