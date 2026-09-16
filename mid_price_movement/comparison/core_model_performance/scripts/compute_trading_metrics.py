import os
import sys
import json
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, f1_score, precision_score, recall_score, confusion_matrix

warnings.filterwarnings("ignore")

BASE_DIR   = Path(__file__).resolve().parent.parent.parent.parent
SCRIPT_DIR = BASE_DIR
OUT_DIR    = Path(__file__).resolve().parent.parent
GRAPHS_DIR = OUT_DIR / "graphs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500]
TICKS_PER_YEAR = 1.2e8

MODEL_CONFIGS = [
    ("Sticker",                 SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n56_bs128_default" / "predictions.parquet"),
    ("Sticker_E56_N32_BS128_Vanilla_Plasticity_0_005_LR_0_001_L2_1e3",  SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n32_bs128_vanilla_plasticity_0_005_lr_0_001_l2_1e3" / "predictions.parquet"),
    ("Sticker_E56_N40_BS128_Vanilla_Plasticity_0_005_LR_0_005_L2_5e4",         SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n40_bs128_vanilla_plasticity_0_005_lr_0_005_l2_5e4" / "predictions.parquet"),
    ("Sticker_E56_N48_BS128_Vanilla_Plasticity_0_005_LR_0_005_L2_5e4",         SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n48_bs128_vanilla_plasticity_0_005_lr_0_005_l2_5e4" / "predictions.parquet"),
    ("Sticker_E56_N48_BS128_Vanilla_Plasticity_0_005_LR_0_001_L2_1e3",  SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n48_bs128_vanilla_plasticity_0_005_lr_0_001_l2_1e3" / "predictions.parquet"),
    ("Sticker_E56_N32_BS32_FastAdapt_Plasticity_0_02_LR_0_003_L2_1e5", SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n32_bs32_fastadapt_plasticity_0_02_lr_0_003_l2_1e5" / "predictions.parquet"),
    ("Sticker_E56_N32_BS32_Vanilla_Plasticity_0_005_LR_0_005_L2_5e4_Seed42",    SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n32_bs32_vanilla_plasticity_0_005_lr_0_005_l2_5e4" / "default" / "predictions.parquet"),
    ("Sticker_E16_N16_BS128_Vanilla_Plasticity_0_0005_LR_0_0005_L2_1e4",         SCRIPT_DIR / "sticker" / "variations" / "sticker_e16_n16_bs128_vanilla_plasticity_0_0005_lr_0_0005_l2_1e4" / "predictions.parquet"),
    ("Sticker_E28_N32_BS128_Vanilla_Plasticity_0_005_LR_0_005_L2_5e4",         SCRIPT_DIR / "sticker" / "variations" / "sticker_e28_n32_bs128_vanilla_plasticity_0_005_lr_0_005_l2_5e4" / "predictions.parquet"),
    ("Sticker_E56_N56_BS128_Vanilla_Plasticity_0_005_LR_0_005_L2_5e4",         SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n56_bs128_vanilla_plasticity_0_005_lr_0_005_l2_5e4" / "predictions.parquet"),
    ("Sticker_E56_N64_BS128_Vanilla_Plasticity_0_005_LR_0_005_L2_5e4",         SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n64_bs128_vanilla_plasticity_0_005_lr_0_005_l2_5e4" / "predictions.parquet"),
    ("Sticker_E56_N112_BS128_Vanilla_Plasticity_0_005_LR_0_005_L2_5e4",        SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n112_bs128_vanilla_plasticity_0_005_lr_0_005_l2_5e4" / "predictions.parquet"),
    ("Sticker_E112_N32_BS128_Vanilla_Plasticity_0_005_LR_0_005_L2_5e4",        SCRIPT_DIR / "sticker" / "variations" / "sticker_e112_n32_bs128_vanilla_plasticity_0_005_lr_0_005_l2_5e4" / "predictions.parquet"),
    ("Sticker_E112_N64_BS128_Vanilla_Plasticity_0_005_LR_0_005_L2_5e4",        SCRIPT_DIR / "sticker" / "variations" / "sticker_e112_n64_bs128_vanilla_plasticity_0_005_lr_0_005_l2_5e4" / "predictions.parquet"),
    ("Sticker_E56_N32_BS256_Vanilla_Plasticity_0_005_LR_0_001_L2_5e4",   SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n32_bs256_vanilla_plasticity_0_005_lr_0_001_l2_5e4" / "predictions.parquet"),
    ("Sticker_E56_N56_BS256_Vanilla_Plasticity_0_005_LR_0_001_L2_5e4",   SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n56_bs256_vanilla_plasticity_0_005_lr_0_001_l2_5e4" / "predictions.parquet"),
    ("Sticker_E56_N32_BS256_Orthogonal_Plasticity_0_005_LR_0_001_L2_5e4", SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n32_bs256_orthogonal_plasticity_0_005_lr_0_001_l2_5e4" / "predictions.parquet"),
    ("Sticker_E56_N32_BS128_Orthogonal_L2_0", SCRIPT_DIR / "sticker" / "sticker_e56_n32_bs128_orthogonal_l2_0" / "predictions.parquet"),
    ("Sticker_E56_N32_BS128_Orthogonal_Plasticity_0_01_L2_0", SCRIPT_DIR / "sticker" / "sticker_e56_n32_bs128_orthogonal_plasticity_0_01_l2_0" / "default" / "predictions.parquet"),
    ("Sticker_E56_N32_BS128_Orthogonal_Plasticity_0_1_L2_0", SCRIPT_DIR / "sticker" / "sticker_e56_n32_bs128_orthogonal_plasticity_0_1_l2_0" / "predictions.parquet"),
    ("Sticker_E56_N32_BS256_Orthogonal_L2_0", SCRIPT_DIR / "sticker" / "sticker_e56_n32_bs256_orthogonal_l2_0" / "predictions.parquet"),
    ("Sticker_E56_N32_BS256_Orthogonal_Plasticity_0_01_L2_0", SCRIPT_DIR / "sticker" / "sticker_e56_n32_bs256_orthogonal_plasticity_0_01_l2_0" / "default" / "predictions.parquet"),
    ("Sticker_E56_N32_BS256_Orthogonal_Plasticity_0_01_L2_0_Seed67", SCRIPT_DIR / "sticker" / "sticker_e56_n32_bs256_orthogonal_plasticity_0_01_l2_0" / "seed_67" / "predictions.parquet"),
    ("Sticker_E56_N32_BS256_Orthogonal_Plasticity_0_01_L2_0_Seed420", SCRIPT_DIR / "sticker" / "sticker_e56_n32_bs256_orthogonal_plasticity_0_01_l2_0" / "seed_420" / "predictions.parquet"),
    ("Sticker_E56_N32_BS256_Orthogonal_Plasticity_0_1_L2_0", SCRIPT_DIR / "sticker" / "sticker_e56_n32_bs256_orthogonal_plasticity_0_1_l2_0" / "predictions.parquet"),
    ("Sticker_E56_N56_BS128_Orthogonal_Plasticity_0_01_L2_0", SCRIPT_DIR / "sticker" / "sticker_e56_n56_bs128_orthogonal_plasticity_0_01_l2_0" / "default" / "predictions.parquet"),
    ("Sticker_E56_N56_BS256_Orthogonal_Plasticity_0_01_L2_0", SCRIPT_DIR / "sticker" / "sticker_e56_n56_bs256_orthogonal_plasticity_0_01_l2_0" / "predictions.parquet"),
    ("Sticker_E56_N32_BS256_Orthogonal_Plasticity_0_01_LR_0_003_L2_0", SCRIPT_DIR / "sticker" / "sticker_e56_n32_bs256_orthogonal_plasticity_0_01_lr_0_003_l2_0" / "predictions.parquet"),
    ("Sticker_E56_N56_BS256_Orthogonal_Plasticity_0_01_LR_0_003_L2_0", SCRIPT_DIR / "sticker" / "sticker_e56_n56_bs256_orthogonal_plasticity_0_01_lr_0_003_l2_0" / "predictions.parquet"),
    ("Sticker_E56_N32_BS128_Orthogonal_Plasticity_0_01_LR_0_001_L2_1e6", SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n32_bs128_orthogonal_plasticity_0_01_lr_0_001_l2_1e6" / "predictions.parquet"),
    ("Sticker_E56_N32_BS256_Orthogonal_Plasticity_0_01_LR_0_001_L2_1e6", SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n32_bs256_orthogonal_plasticity_0_01_lr_0_001_l2_1e6" / "predictions.parquet"),
    ("Sticker_E56_N48_BS256_Orthogonal_Plasticity_0_01_LR_0_001_L2_0", SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n48_bs256_orthogonal_plasticity_0_01_lr_0_001_l2_0" / "predictions.parquet"),
    ("Sticker_E56_N32_BS256_Orthogonal_Plasticity_0_01_LR_0_001_L2_5e6", SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n32_bs256_orthogonal_plasticity_0_01_lr_0_001_l2_5e6" / "predictions.parquet"),
    ("Sticker_E56_N32_BS256_Orthogonal_Plasticity_0_01_LR_0_001_L2_1e5", SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n32_bs256_orthogonal_plasticity_0_01_lr_0_001_l2_1e5" / "predictions.parquet"),
    ("Sticker_E56_N32_BS256_Orthogonal_Plasticity_0_01_LR_0_001_L2_5e5", SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n32_bs256_orthogonal_plasticity_0_01_lr_0_001_l2_5e5" / "predictions.parquet"),
    ("Sticker_E56_N32_BS256_Orthogonal_Plasticity_0_1_LR_0_001_L2_1e6", SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n32_bs256_orthogonal_plasticity_0_1_lr_0_001_l2_1e6" / "predictions.parquet"),
    ("Sticker_E56_N32_BS256_Orthogonal_Plasticity_0_01_LR_0_0005_L2_0", SCRIPT_DIR / "sticker" / "sticker_e56_n32_bs256_orthogonal_plasticity_0_01_lr_0_0005_l2_0" / "predictions.parquet"),
    ("Sticker_E56_N56_BS256_Orthogonal_Plasticity_0_01_LR_0_0005_L2_0", SCRIPT_DIR / "sticker" / "sticker_e56_n56_bs256_orthogonal_plasticity_0_01_lr_0_0005_l2_0" / "default" / "predictions.parquet"),
    ("Sticker_E56_N56_BS256_Orthogonal_Plasticity_0_01_LR_0_0005_L2_0_Seed67", SCRIPT_DIR / "sticker" / "sticker_e56_n56_bs256_orthogonal_plasticity_0_01_lr_0_0005_l2_0" / "seed_67" / "predictions.parquet"),
    ("Sticker_E56_N56_BS256_Orthogonal_Plasticity_0_01_LR_0_0005_L2_0_Seed420", SCRIPT_DIR / "sticker" / "sticker_e56_n56_bs256_orthogonal_plasticity_0_01_lr_0_0005_l2_0" / "seed_420" / "predictions.parquet"),
    ("Sticker_E56_N56_BS128_Orthogonal_Plasticity_0_01_L2_0_Seed67", SCRIPT_DIR / "sticker" / "sticker_e56_n56_bs128_orthogonal_plasticity_0_01_l2_0" / "seed_67" / "predictions.parquet"),
    ("Sticker_E56_N56_BS128_Orthogonal_Plasticity_0_01_L2_0_Seed420", SCRIPT_DIR / "sticker" / "sticker_e56_n56_bs128_orthogonal_plasticity_0_01_l2_0" / "seed_420" / "predictions.parquet"),
    ("Sticker_E56_N32_BS128_Orthogonal_Plasticity_0_01_L2_0_Seed67", SCRIPT_DIR / "sticker" / "sticker_e56_n32_bs128_orthogonal_plasticity_0_01_l2_0" / "seed_67" / "predictions.parquet"),
    ("Sticker_E56_N32_BS128_Orthogonal_Plasticity_0_01_L2_0_Seed420", SCRIPT_DIR / "sticker" / "sticker_e56_n32_bs128_orthogonal_plasticity_0_01_l2_0" / "seed_420" / "predictions.parquet"),
    ("Sticker_E56_N56_BS128_Orthogonal_Plasticity_0_1_L2_0", SCRIPT_DIR / "sticker" / "sticker_e56_n56_bs128_orthogonal_plasticity_0_1_l2_0" / "predictions.parquet"),
    ("Sticker_E56_N56_BS128_Orthogonal_Plasticity_0_05_L2_0", SCRIPT_DIR / "sticker" / "sticker_e56_n56_bs128_orthogonal_plasticity_0_05_l2_0" / "predictions.parquet"),
    ("Sticker_E56_N32_BS128_Orthogonal_Plasticity_0_01_LR_0_0005_L2_0", SCRIPT_DIR / "sticker" / "sticker_e56_n32_bs128_orthogonal_plasticity_0_01_lr_0_0005_l2_0" / "default" / "predictions.parquet"),
    ("Sticker_E56_N32_BS128_Orthogonal_Plasticity_0_01_LR_0_0005_L2_0_Seed67", SCRIPT_DIR / "sticker" / "sticker_e56_n32_bs128_orthogonal_plasticity_0_01_lr_0_0005_l2_0" / "seed_67" / "predictions.parquet"),
    ("Sticker_E56_N32_BS128_Orthogonal_Plasticity_0_01_LR_0_0005_L2_0_Seed420", SCRIPT_DIR / "sticker" / "sticker_e56_n32_bs128_orthogonal_plasticity_0_01_lr_0_0005_l2_0" / "seed_420" / "predictions.parquet"),
    ("Sticker_E56_N56_BS128_Orthogonal_Plasticity_0_01_L2_0_Seed1", SCRIPT_DIR / "sticker" / "sticker_e56_n56_bs128_orthogonal_plasticity_0_01_l2_0" / "seed_1" / "predictions.parquet"),
    ("Sticker_E56_N56_BS128_Orthogonal_Plasticity_0_01_L2_0_Seed999", SCRIPT_DIR / "sticker" / "sticker_e56_n56_bs128_orthogonal_plasticity_0_01_l2_0" / "seed_999" / "predictions.parquet"),
    ("Sticker_E56_N32_BS128_Orthogonal_Plasticity_0_01_L2_0_Seed999", SCRIPT_DIR / "sticker" / "sticker_e56_n32_bs128_orthogonal_plasticity_0_01_l2_0" / "seed_999" / "predictions.parquet"),
    ("Sticker_E56_N56_BS256_Orthogonal_Plasticity_0_01_LR_0_0005_L2_0_Seed1", SCRIPT_DIR / "sticker" / "sticker_e56_n56_bs256_orthogonal_plasticity_0_01_lr_0_0005_l2_0" / "seed_1" / "predictions.parquet"),
    ("Sticker_E56_N32_BS256_Orthogonal_Plasticity_0_01_L2_0_Seed1", SCRIPT_DIR / "sticker" / "sticker_e56_n32_bs256_orthogonal_plasticity_0_01_l2_0" / "seed_1" / "predictions.parquet"),
    ("Sticker_E56_N56_BS128_Orthogonal_Plasticity_0_01_LR_0_0005_L2_0", SCRIPT_DIR / "sticker" / "sticker_e56_n56_bs128_orthogonal_plasticity_0_01_lr_0_0005_l2_0" / "predictions.parquet"),
    ("Sticker_E56_N56_BS256_Orthogonal_Plasticity_0_01_LR_0_00001_L2_0", SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n56_bs256_orthogonal_plasticity_0_01_lr_0_00001_l2_0" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_Plasticity_0_01_L2_0_Seed67", SCRIPT_DIR / "sticker" / "sticker_e56_n32_orthogonal_plasticity_0_01_l2_0" / "seed_67" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_Plasticity_0_01_L2_0_Seed420", SCRIPT_DIR / "sticker" / "sticker_e56_n32_orthogonal_plasticity_0_01_l2_0" / "seed_420" / "predictions.parquet"),
    ("Sticker_E56_N56_BS256_Orthogonal_Plasticity_0_005_LR_0_001_L2_5e4", SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n56_bs256_orthogonal_plasticity_0_005_lr_0_001_l2_5e4" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_Plasticity_0_005_LR_0_001_L2_5e4",   SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n32_orthogonal_plasticity_0_005_lr_0_001_l2_5e4" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_Plasticity_0_005_LR_0_001_L2_1e3", SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n32_orthogonal_plasticity_0_005_lr_0_001_l2_1e3" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_Plasticity_0_005_LR_0_001_L2_1e4", SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n32_orthogonal_plasticity_0_005_lr_0_001_l2_1e4" / "predictions.parquet"),
    ("Sticker_E56_N48_Orthogonal_Plasticity_0_005_LR_0_001_L2_5e4", SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n48_orthogonal_plasticity_0_005_lr_0_001_l2_5e4" / "predictions.parquet"),
    ("Sticker_E56_N56_Orthogonal_Plasticity_0_005_LR_0_001_L2_5e4", SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n56_orthogonal_plasticity_0_005_lr_0_001_l2_5e4" / "predictions.parquet"),
    ("Sticker_E56_N56_Orthogonal_Plasticity_0_005_LR_0_001_L2_1e4", SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n56_orthogonal_plasticity_0_005_lr_0_001_l2_1e4" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_Plasticity_1e3_LR_0_001_L2_5e4", SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n32_orthogonal_plasticity_1e3_lr_0_001_l2_5e4" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_Plasticity_0_005_LR_0_001_L2_1e5", SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n32_orthogonal_plasticity_0_005_lr_0_001_l2_1e5" / "predictions.parquet"),
    ("Sticker_E56_N56_Orthogonal_Plasticity_0_005_LR_0_001_L2_1e5", SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n56_orthogonal_plasticity_0_005_lr_0_001_l2_1e5" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_Plasticity_0_005_LR_0_001_L2_1e6", SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n32_orthogonal_plasticity_0_005_lr_0_001_l2_1e6" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_Plasticity_0_005_LR_0_001_L2_1e7", SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n32_orthogonal_plasticity_0_005_lr_0_001_l2_1e7" / "predictions.parquet"),
    ("Sticker_E56_N56_Orthogonal_Plasticity_0_005_LR_0_001_L2_1e6", SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n56_orthogonal_plasticity_0_005_lr_0_001_l2_1e6" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_Plasticity_0_005_LR_0_001_L2_1e10", SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n32_orthogonal_plasticity_0_005_lr_0_001_l2_1e10" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_L2_0", SCRIPT_DIR / "sticker" / "sticker_e56_n32_orthogonal_l2_0" / "default" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_Plasticity_0_01_L2_0", SCRIPT_DIR / "sticker" / "sticker_e56_n32_orthogonal_plasticity_0_01_l2_0" / "default" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_Plasticity_0_05_LR_0_001_L2_0", SCRIPT_DIR / "sticker" / "sticker_e56_n32_orthogonal_plasticity_0_05_lr_0_001_l2_0" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_Plasticity_0_1_LR_0_001_L2_0",  SCRIPT_DIR / "sticker" / "sticker_e56_n32_orthogonal_plasticity_0_1_lr_0_001_l2_0" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_L2_0_Seed1337", SCRIPT_DIR / "sticker" / "sticker_e56_n32_orthogonal_l2_0" / "seed_1337" / "predictions.parquet"),
    ("Sticker_E56_N56_Orthogonal_Plasticity_0_005_LR_0_001_L2_0", SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n56_orthogonal_plasticity_0_005_lr_0_001_l2_0" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_PurePlasticity_LR_0_001_L2_0", SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n32_orthogonal_pure_plasticity_lr_0_001_l2_0" / "predictions.parquet"),
    ("Sticker_E56_N32_BS32_Vanilla_Plasticity_0_005_LR_0_005_L2_5e4_Seed1337",   SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n32_bs32_vanilla_plasticity_0_005_lr_0_005_l2_5e4" / "seed_1337" / "predictions.parquet"),
    ("Logistic Regression",             SCRIPT_DIR / "baselines" / "logreg" / "predictions.parquet"),
    ("Kalman Filter",        SCRIPT_DIR / "baselines" / "kalman" / "predictions.parquet"),
    ("ARIMA",             SCRIPT_DIR / "baselines" / "arima" / "predictions.parquet"),
    ("OFI Linear Regression",           SCRIPT_DIR / "baselines" / "ofi_linreg" / "predictions.parquet"),
    ("GARCH(1,1)",              SCRIPT_DIR / "baselines" / "garch" / "predictions.parquet"),
    ("LightGBM",                SCRIPT_DIR / "baselines" / "lgb" / "predictions.parquet"),
    ("DeepLOB",             SCRIPT_DIR / "baselines" / "deeplob" / "predictions.parquet"),
    ("Mamba",                      SCRIPT_DIR / "baselines" / "mamba" / "predictions.parquet"),
    ("TLOB",         SCRIPT_DIR / "baselines" / "tlob" / "predictions.parquet"),
]


def compute_model_metrics(name: str, pred_path: Path) -> dict:
    if not pred_path.exists():
        print(f"[SKIP] {name}: {pred_path} not found.")
        return None

    t0 = time.time()
    print(f"[PROCESS] {name} ...", flush=True)
    df = pd.read_parquet(pred_path).sort_values("tick_idx").reset_index(drop=True)
    df["mid"] = df["mid"].astype(float)
    df["pred"] = df["pred"].astype(int)

    total_ticks = len(df)
    n_75 = total_ticks // 4
    df = df.iloc[n_75:].reset_index(drop=True)

    n_samples = len(df)
    mids = df["mid"].values
    preds = df["pred"].values

    lat_p50 = float(df["lat_ms"].median()) if "lat_ms" in df.columns else None
    lat_p99 = float(df["lat_ms"].quantile(0.99)) if "lat_ms" in df.columns else None

    mid_diff_h1 = np.diff(mids, append=np.nan)
    ret_bps_h1 = (mid_diff_h1 / mids) * 10000.0
    actual_labels_h1 = np.where(mid_diff_h1 > 1e-8, 1, np.where(mid_diff_h1 < -1e-8, -1, 0))

    valid_mask = ~np.isnan(mid_diff_h1) & (actual_labels_h1 != 0)
    actual_nz = actual_labels_h1[valid_mask]
    preds_nz = preds[valid_mask]

    mcc = float(matthews_corrcoef(actual_nz, preds_nz))
    f1_macro = float(f1_score(actual_nz, preds_nz, average="macro"))
    f1_weighted = float(f1_score(actual_nz, preds_nz, average="weighted"))
    precision = float(precision_score(actual_nz, preds_nz, average="weighted", zero_division=0))
    recall = float(recall_score(actual_nz, preds_nz, average="weighted", zero_division=0))
    da_nz_h1 = float(np.mean(actual_nz == preds_nz))

    cm = confusion_matrix(actual_nz, preds_nz, labels=[-1, 1])

    pnl_gross_bps = preds[:-1] * ret_bps_h1[:-1]
    cum_pnl_bps = np.cumsum(np.nan_to_num(pnl_gross_bps))

    total_gross_pnl_bps = float(np.sum(pnl_gross_bps))
    mean_edge_bps = float(np.mean(pnl_gross_bps))
    std_edge_bps = float(np.std(pnl_gross_bps))

    pos_pnl = pnl_gross_bps[pnl_gross_bps > 0]
    neg_pnl = pnl_gross_bps[pnl_gross_bps < 0]

    win_rate = float(len(pos_pnl) / (len(pos_pnl) + len(neg_pnl))) if (len(pos_pnl) + len(neg_pnl)) > 0 else 0.0
    profit_factor = float(np.sum(pos_pnl) / abs(np.sum(neg_pnl))) if len(neg_pnl) > 0 and np.sum(neg_pnl) != 0 else np.nan

    ann_sharpe = float((mean_edge_bps / std_edge_bps) * np.sqrt(TICKS_PER_YEAR)) if std_edge_bps > 0 else 0.0
    downside_std = float(np.std(pnl_gross_bps[pnl_gross_bps < 0])) if len(neg_pnl) > 0 else 1.0
    ann_sortino = float((mean_edge_bps / downside_std) * np.sqrt(TICKS_PER_YEAR)) if downside_std > 0 else 0.0

    running_max = np.maximum.accumulate(cum_pnl_bps)
    drawdowns = running_max - cum_pnl_bps
    max_drawdown_bps = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    horizon_da_nz = {}
    for h in HORIZONS:
        m_diff = pd.Series(mids).shift(-h).values - mids
        actual_h = np.where(m_diff > 1e-8, 1, np.where(m_diff < -1e-8, -1, 0))
        m_nz = ~np.isnan(m_diff) & (actual_h != 0)
        da_h = float(np.mean(actual_h[m_nz] == preds[m_nz])) if m_nz.sum() > 0 else None
        horizon_da_nz[f"h{h}"] = da_h

    elapsed = time.time() - t0
    print(f"[DONE] {name} in {elapsed:.1f}s | h1 DA: {da_nz_h1:.4f} | Sharpe: {ann_sharpe:.1f}", flush=True)

    return {
        "model_name": name,
        "eval_ticks": n_samples,
        "da_nz_h1": round(da_nz_h1, 4),
        "mcc_h1": round(mcc, 4),
        "f1_macro_h1": round(f1_macro, 4),
        "f1_weighted_h1": round(f1_weighted, 4),
        "precision_h1": round(precision, 4),
        "recall_h1": round(recall, 4),
        "total_gross_pnl_bps": round(total_gross_pnl_bps, 2),
        "mean_edge_per_tick_bps": round(mean_edge_bps, 5),
        "profit_factor": round(profit_factor, 2) if not np.isnan(profit_factor) else None,
        "win_rate_pct": round(win_rate * 100.0, 2),
        "annualized_sharpe": round(ann_sharpe, 1),
        "annualized_sortino": round(ann_sortino, 1),
        "max_drawdown_bps": round(max_drawdown_bps, 2),
        "latency_p50_ms": round(lat_p50, 4) if lat_p50 is not None else None,
        "latency_p99_ms": round(lat_p99, 4) if lat_p99 is not None else None,
        "confusion_matrix_h1": cm.tolist(),
        "horizon_da_nz": horizon_da_nz,
    }


def main():
    print("=" * 85, flush=True)
    print("  ACCURACY & FINANCIAL METRICS - LAST 75% TICKS EVALUATION")
    print("=" * 85, flush=True)

    results = []
    for name, path in MODEL_CONFIGS:
        m = compute_model_metrics(name, path)
        if m is not None:
            results.append(m)

    json_path = OUT_DIR / "test_metrics.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    df_res = pd.DataFrame(results)
    csv_path = OUT_DIR / "test_rank.csv"
    df_res.sort_values("da_nz_h1", ascending=False).to_csv(csv_path, index=False)

    print(f"\n[SAVED] {json_path}")
    print(f"[SAVED] {csv_path}", flush=True)

    generate_markdown_leaderboard(results)

    print("\n" + "=" * 85, flush=True)
    print("  COMPLETE!")
    print(f"  Leaderboard: {OUT_DIR / 'README.md'}", flush=True)
    print("=" * 85, flush=True)


def generate_markdown_leaderboard(results: list[dict]):
    df = pd.DataFrame(results).sort_values("da_nz_h1", ascending=False).reset_index(drop=True)

    md_lines = [
        "# Accuracy & Financial Performance Leaderboard",
        "",
        "**Dataset**: 11,918,929 Real-Time Trade Ticks (Hyperliquid BTC Perpetual Futures, Dec 1-31, 2025)  ",
        "**Evaluation Basis**: Last 75% of ticks (prequential stream), non-zero mid-price movement ticks  ",
        "**Protocol**: Zero-Lookahead Prequential (Test-Then-Train) Online Streaming  ",
        "",
        "---",
        "",
        "## 1. Master Financial & Execution Metric Leaderboard (h=1)",
        "",
        "| Model | Non-Zero DA | MCC | Weighted F1 | Signal Edge (bps/tick) | Profit Factor | Win Rate (%) | Ann. Sharpe | Ann. Sortino | Max DD (bps) | Latency p99 (ms) |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for _, row in df.iterrows():
        name = row['model_name']
        da   = f"{row['da_nz_h1']:.4f}"
        mcc  = f"{row['mcc_h1']:.4f}"
        f1   = f"{row['f1_weighted_h1']:.4f}"
        edge = f"{row['mean_edge_per_tick_bps']:.4f}"
        pf   = f"{row['profit_factor']:.2f}" if row['profit_factor'] is not None else "N/A"
        wr   = f"{row['win_rate_pct']:.1f}%"
        shp  = f"{row['annualized_sharpe']:,.1f}"
        srt  = f"{row['annualized_sortino']:,.1f}"
        mdd  = f"{row['max_drawdown_bps']:.1f}"
        lat  = f"{row['latency_p99_ms']:.2f}" if row['latency_p99_ms'] is not None else "N/A"
        md_lines.append(f"| **{name}** | **{da}** | {mcc} | {f1} | {edge} | {pf} | {wr} | {shp} | {srt} | {mdd} | {lat} |")

    md_lines.extend([
        "",
        "---",
        "",
        "## 2. Multi-Horizon Non-Zero Directional Accuracy (Last 75% Ticks)",
        "",
        "| Horizon h | " + " | ".join(r["model_name"][:30] for r in results[:20]) + " |",
        "|---:| " + ":---:| " * min(20, len(results)),
    ])

    model_lookup = {r["model_name"]: r["horizon_da_nz"] for r in results}

    for h in HORIZONS:
        row_str = f"| **h={h}** "
        for r in results[:20]:
            m_name = r["model_name"]
            if m_name in model_lookup and f"h{h}" in model_lookup[m_name]:
                val = model_lookup[m_name][f"h{h}"]
                row_str += f"| {val:.4f} " if val is not None else "| N/A "
            else:
                row_str += "| N/A "
        row_str += "|"
        md_lines.append(row_str)

    report_path = OUT_DIR / "README.md"
    with open(report_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")


if __name__ == "__main__":
    main()
