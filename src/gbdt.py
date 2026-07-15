# %%
import sys

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from hydra import compose, initialize
import os
from sklearn.preprocessing import OrdinalEncoder
import rootutils
import wandb
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
import copy
from tqdm import tqdm

from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import VotingClassifier

from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

import lightgbm as lgb
import catboost as cb
import xgboost as xgb
import optuna

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.isic_utils.utils import (
    prepare_df_for_gbdt,
    comp_score,
    custom_metric,
    preprocess_df,
    SelectColumns,
    DNNFeatureEngineering,
)
from src.isic_utils.feature_engineering import feature_engineering_new
from src.isic_utils.gbdt_models import GBDTModels

DEFAULT_GBDT_PARAMS = "260709-0NNs-18types-feV7-s5-tuning_weights"


def get_folds_to_run(cfg):
    cv_fold = cfg.get("cv_fold", None)
    if cv_fold is None:
        return list(range(cfg.data.n_fold))
    fold = int(cv_fold)
    if fold < 0 or fold >= cfg.data.n_fold:
        raise ValueError(f"cv_fold must be in [0, {cfg.data.n_fold - 1}], got {fold}")
    return [fold]


def require_all_fold_models(cfg):
    missing = [
        fold
        for fold in range(cfg.data.n_fold)
        if not os.path.exists(os.path.join(cfg.log_dir, f"model_{fold}.joblib"))
    ]
    if missing:
        raise FileNotFoundError(f"Missing fold checkpoints: {missing} in {cfg.log_dir}")


# %%
gbdt_params = DEFAULT_GBDT_PARAMS
cli_overrides = [a for a in sys.argv[1:] if "=" in a]
if not any(o.split("=", 1)[0] == "gbdt_params" for o in cli_overrides):
    cli_overrides.insert(0, f"gbdt_params={gbdt_params}")

with initialize(version_base=None, config_path="../configs"):
    cfg = compose(
        config_name="gbdt",
        overrides=cli_overrides,
        return_hydra_config=True,
    )
    cfg.paths.output_dir = "${hydra.runtime.output_dir}"
    cfg.paths.work_dir = "${hydra.runtime.cwd}"
    cfg.hydra.run.dir = cfg.log_dir
    cfg.hydra.runtime.output_dir = cfg.hydra.run.dir

os.makedirs(cfg.log_dir, exist_ok=True)

# %%
df_train = pd.read_csv(os.path.join(cfg.data.data_dir, cfg.data.meta_csv_train_name))
df_test = pd.read_csv(os.path.join(cfg.data.data_dir, cfg.data.meta_csv_test_name))
df_fold = pd.read_parquet(os.path.join(cfg.data.data_dir, cfg.data.kfold_df_name))
assert (df_train["isic_id"] == df_fold["isic_id"]).all()

folds = []
for k in range(cfg.data.n_fold):
    if cfg.data.kfold_method == "sgkf":
        col_name = f"StratifiedGroupKFold_{cfg.data.n_fold}_{k}"
    if cfg.data.kfold_method == "tsgkf":
        col_name = f"TSGKF_{cfg.data.n_fold}_{k}"
    else:
        assert False
    train_idx = np.where(df_fold[col_name] == "train")[0]
    val_idx = np.where(df_fold[col_name] == "val")[0]
    folds.append([train_idx, val_idx])


# %%
if cfg.gbdt_params.use_logits:
    dnn_col_name = "logits"
else:
    dnn_col_name = "probabilities"

dnn_run_name_list = []
df_dnn_preds_list = []
for dnn_run in cfg.gbdt_params.get("dnn_predictions", []):
    df_list = []
    for k in range(cfg.data.n_fold):
        df_tmp = pd.read_parquet(dnn_run.dir + f"/fold{k}.parquet")
        df_tmp = df_tmp[["isic_id", dnn_col_name]].rename({dnn_col_name: "predictions"}, axis="columns")
        if cfg.gbdt_params.dnn_binning:
            assert not cfg.gbdt_params.use_logits
            bins = np.linspace(0, 1, cfg.gbdt_params.dnn_binning)
            df_tmp["predictions"] = pd.cut(df_tmp["predictions"], bins=bins, labels=False)
        df_list.append(df_tmp)
    df_preds = df_list[0]
    for k, _df_preds in enumerate(df_list[1:], start=1):
        df_preds = df_preds.merge(_df_preds, how="left", on="isic_id", suffixes=("", f"_{k}"))
    df_preds = df_preds.rename({"predictions": "predictions_0"}, axis="columns")
    df_dnn_preds_list.append(df_preds)
    dnn_run_name_list.append(dnn_run.name)

# stackingのために、val予測を新たなtrainとする
df_train_2nd = []
for fold in range(cfg.data.n_fold):
    for run_name, df_preds in zip(dnn_run_name_list, df_dnn_preds_list):
        assert (df_train["isic_id"] == df_preds["isic_id"]).all()
        df_train[run_name] = df_preds[f"predictions_{fold}"]

    _df_valid = df_train.iloc[folds[fold][1]]
    df_train_2nd.append(_df_valid)

df_train = pd.concat(df_train_2nd).sort_index()

# dummy
df_test[dnn_run_name_list] = 0

# %%
fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(
    df_train[dnn_run_name_list].corr(), square=True, vmax=1, vmin=-1, center=0, cmap="coolwarm", annot=True
)

# %%
df_train, feature_cols, cat_cols = feature_engineering_new(df_train, version=cfg.gbdt_params.version_fe)
df_test, _, _ = feature_engineering_new(df_test, version=cfg.gbdt_params.version_fe)

df_train, df_test, feature_cols, cat_cols = preprocess_df(df_train, df_test, feature_cols, cat_cols)
target_col = "target"

feature_cols_without_dnn = copy.copy(feature_cols)
feature_cols += dnn_run_name_list

# %%
result_dict = {}
best_weights = None

if cfg.gbdt_stage == "fold":
    for fold in get_folds_to_run(cfg):
        save_file_path = os.path.join(cfg.log_dir, f"model_{fold}.joblib")
        if os.path.exists(save_file_path):
            print(f"fold: {fold} - skip (checkpoint exists: {save_file_path})")
            continue

        _df_train = df_train.iloc[folds[fold][0]].reset_index(drop=True)
        _df_valid = df_train.iloc[folds[fold][1]].reset_index(drop=True)

        X, y = _df_train, _df_train[target_col]

        gbdt_models = GBDTModels(cfg.gbdt_params, feature_cols_without_dnn, cat_cols)
        gbdt_models.fit(X, y)
        joblib.dump(gbdt_models, save_file_path)

        preds = gbdt_models.predict(_df_valid)
        score = comp_score(_df_valid[[target_col]], pd.DataFrame(preds, columns=["prediction"]), "")
        print(f"fold: {fold} - Partial AUC Score: {score:.5f}")

        preds_all = gbdt_models.predict(df_train)
        df_preds = pd.DataFrame({"isic_id": df_train["isic_id"], "predictions": preds_all})
        df_preds.to_parquet(os.path.join(cfg.log_dir, f"fold{fold}.parquet"))

        result_dict[f"fold_{fold}"] = score

    if result_dict:
        result_dict["cv_score"] = np.mean(list(result_dict.values()))
        print(result_dict)

# %% tuning ensemble weights
if cfg.gbdt_stage == "tune" and cfg.gbdt_params.get("tuning_ensemble_weights"):
    require_all_fold_models(cfg)
    num_models = len(cfg.gbdt_params.models)

    val_preds = []
    val_targets = []

    if cfg.gbdt_params.get("tuning_4folds"):
        fold_list = range(1, cfg.data.n_fold)
    else:
        fold_list = range(cfg.data.n_fold)

    for fold in fold_list:
        _df_valid = df_train.iloc[folds[fold][1]].reset_index(drop=True)
        save_file_path = os.path.join(cfg.log_dir, f"model_{fold}.joblib")
        gbdt_models = joblib.load(save_file_path)

        preds = gbdt_models.predict_(_df_valid)
        targets = _df_valid[target_col]

        val_preds.append(preds)
        val_targets.append(targets)

    def objective(trial):
        w_list = []
        for i in range(num_models):
            w = trial.suggest_float(f"w_{i}", 0, 1)
            w_list.append(w)
        w_list = np.array(w_list)

        # 重みの正規化
        w_list = w_list / w_list.sum()

        results = []
        for fold in range(len(fold_list)):
            preds = np.average(val_preds[fold], axis=0, weights=w_list)
            targets = val_targets[fold]
            score = comp_score(pd.DataFrame(targets), pd.DataFrame(preds, columns=["prediction"]), "")
            results.append(score)

        return np.array(results).mean()

    # Optunaによる最適化
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1000)

    # 最適な重み
    best_weights = study.best_params
    print("Best weights:", best_weights)

    best_weights = np.array(list(best_weights.values()))

# %%
if cfg.gbdt_stage == "tune" and cfg.gbdt_params.get("tuning_ensemble_weights"):
    result_dict = {}
    for fold in range(cfg.data.n_fold):
        _df_valid = df_train.iloc[folds[fold][1]].reset_index(drop=True)

        save_file_path = os.path.join(cfg.log_dir, f"model_{fold}.joblib")
        gbdt_models = joblib.load(save_file_path)

        gbdt_models.set_ensemble_weights(best_weights)
        joblib.dump(gbdt_models, save_file_path)

        preds = gbdt_models.predict(_df_valid)
        score = comp_score(_df_valid[[target_col]], pd.DataFrame(preds, columns=["prediction"]), "")
        print(f"fold: {fold} - Partial AUC Score: {score:.5f}")

        # save predictions for stacking
        preds_all = gbdt_models.predict(df_train)
        df_preds = pd.DataFrame({"isic_id": df_train["isic_id"], "predictions": preds_all})
        df_preds.to_parquet(os.path.join(cfg.log_dir, f"fold{fold}.parquet"))

        result_dict[f"fold_{fold}"] = score

    result_dict["cv_score"] = np.array(
        [result_dict[f"fold_{fold}"] for fold in range(cfg.data.n_fold)]
    ).mean()

    print(result_dict)

    run = wandb.init(
        project=cfg.logger.wandb.project,
        entity=cfg.logger.wandb.get("entity"),
        name=f"{cfg.gbdt_params.name}",
        dir=cfg.wandb_summary_dir,
        config=OmegaConf.to_container(cfg.gbdt_params, resolve=True, throw_on_missing=True),
    )
    run.log(result_dict)
    run.finish()

# %% train with all data
if cfg.gbdt_stage == "all_data":
    if cfg.gbdt_params.get("tuning_ensemble_weights"):
        require_all_fold_models(cfg)
        tuned_model = joblib.load(os.path.join(cfg.log_dir, "model_0.joblib"))
        if tuned_model.ensemble_weights is None:
            raise ValueError("Run gbdt_stage=tune before gbdt_stage=all_data")
        best_weights = tuned_model.ensemble_weights

    X, y = df_train, df_train[target_col]

    gbdt_models = GBDTModels(cfg.gbdt_params, feature_cols_without_dnn, cat_cols)
    gbdt_models.fit(X, y)

    if cfg.gbdt_params.get("tuning_ensemble_weights"):
        gbdt_models.set_ensemble_weights(best_weights)

    save_file_path = os.path.join(cfg.log_dir, "model_all_data.joblib")
    joblib.dump(gbdt_models, save_file_path)
    print(f"saved: {save_file_path}")

# %% feature importance
if cfg.gbdt_stage == "feat_imp":
    require_all_fold_models(cfg)

fi = []
if cfg.gbdt_stage == "feat_imp":
    for fold in range(cfg.data.n_fold):
        save_file_path = os.path.join(cfg.log_dir, f"model_{fold}.joblib")
        gbdt_models = joblib.load(save_file_path)

        fi.append(gbdt_models.get_feature_importance())

    for fold in range(cfg.data.n_fold):
        for model_idx in range(len(fi[fold][0])):
            df_imp = (
                pd.DataFrame({"feature": fi[fold][1][model_idx], "importance": fi[fold][0][model_idx]})
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )
            df_imp.to_csv(os.path.join(cfg.log_dir, f"feat_imp-fold{fold}-{model_idx}.csv"), index=False)

            # plt.figure(figsize=(16, 12))
            # plt.barh(df_imp["feature"], df_imp["importance"])
            # plt.show()

# %%
