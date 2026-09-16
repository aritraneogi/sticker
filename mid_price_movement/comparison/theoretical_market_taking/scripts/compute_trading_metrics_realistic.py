import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    matthews_corrcoef, f1_score, precision_score, recall_score, confusion_matrix,
)

warnings.filterwarnings("ignore")

BASE_DIR   = Path(__file__).resolve().parent.parent.parent.parent
SCRIPT_DIR = BASE_DIR
OUT_DIR    = Path(__file__).resolve().parent.parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

TAKER_FEE_BPS   = 2.5
HALF_SPREAD_BPS = 0.5
LAT_SLIP_BPS    = 0.5
COST_PER_SIDE   = TAKER_FEE_BPS + HALF_SPREAD_BPS + LAT_SLIP_BPS

COST_SCENARIOS: dict = {
    "zero_cost":    0.0,
    "optimistic":   4.0,
    "realistic":    7.0,
    "conservative": 12.0,
    "pessimistic":  18.0,
}
DEFAULT_RT_BPS = COST_SCENARIOS["realistic"]

NOTIONAL_USD = 100_000
DATA_DAYS    = 31.0

HORIZONS = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500]

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
    ("Sticker_E56_N56_BS256_Orthogonal_Plasticity_0_005_LR_0_001_L2_5e4",   SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n56_bs256_orthogonal_plasticity_0_005_lr_0_001_l2_5e4" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_Plasticity_0_005_LR_0_001_L2_5e4",     SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n32_orthogonal_plasticity_0_005_lr_0_001_l2_5e4" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_Plasticity_0_005_LR_0_001_L2_1e3", SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n32_orthogonal_plasticity_0_005_lr_0_001_l2_1e3" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_Plasticity_0_005_LR_0_001_L2_1e4", SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n32_orthogonal_plasticity_0_005_lr_0_001_l2_1e4" / "predictions.parquet"),
    ("Sticker_E56_N48_Orthogonal_Plasticity_0_005_LR_0_001_L2_5e4",     SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n48_orthogonal_plasticity_0_005_lr_0_001_l2_5e4" / "predictions.parquet"),
    ("Sticker_E56_N56_Orthogonal_Plasticity_0_005_LR_0_001_L2_5e4",     SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n56_orthogonal_plasticity_0_005_lr_0_001_l2_5e4" / "predictions.parquet"),
    ("Sticker_E56_N56_Orthogonal_Plasticity_0_005_LR_0_001_L2_1e4", SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n56_orthogonal_plasticity_0_005_lr_0_001_l2_1e4" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_Plasticity_1e3_LR_0_001_L2_5e4", SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n32_orthogonal_plasticity_1e3_lr_0_001_l2_5e4" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_Plasticity_0_005_LR_0_001_L2_1e5", SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n32_orthogonal_plasticity_0_005_lr_0_001_l2_1e5" / "predictions.parquet"),
    ("Sticker_E56_N56_Orthogonal_Plasticity_0_005_LR_0_001_L2_1e5", SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n56_orthogonal_plasticity_0_005_lr_0_001_l2_1e5" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_Plasticity_0_005_LR_0_001_L2_1e6", SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n32_orthogonal_plasticity_0_005_lr_0_001_l2_1e6" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_Plasticity_0_005_LR_0_001_L2_1e7", SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n32_orthogonal_plasticity_0_005_lr_0_001_l2_1e7" / "predictions.parquet"),
    ("Sticker_E56_N56_Orthogonal_Plasticity_0_005_LR_0_001_L2_1e6", SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n56_orthogonal_plasticity_0_005_lr_0_001_l2_1e6" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_Plasticity_0_005_LR_0_001_L2_1e10", SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n32_orthogonal_plasticity_0_005_lr_0_001_l2_1e10" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_L2_0",    SCRIPT_DIR / "sticker" / "sticker_e56_n32_orthogonal_l2_0" / "default" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_Plasticity_0_01_L2_0", SCRIPT_DIR / "sticker" / "sticker_e56_n32_orthogonal_plasticity_0_01_l2_0" / "default" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_Plasticity_0_05_LR_0_001_L2_0", SCRIPT_DIR / "sticker" / "sticker_e56_n32_orthogonal_plasticity_0_05_lr_0_001_l2_0" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_Plasticity_0_1_LR_0_001_L2_0",  SCRIPT_DIR / "sticker" / "sticker_e56_n32_orthogonal_plasticity_0_1_lr_0_001_l2_0" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_L2_0_Seed1337",   SCRIPT_DIR / "sticker" / "sticker_e56_n32_orthogonal_l2_0" / "seed_1337" / "predictions.parquet"),
    ("Sticker_E56_N56_Orthogonal_Plasticity_0_005_LR_0_001_L2_0",            SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n56_orthogonal_plasticity_0_005_lr_0_001_l2_0" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_PurePlasticity_LR_0_001_L2_0",   SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n32_orthogonal_pure_plasticity_lr_0_001_l2_0" / "predictions.parquet"),
    ("Sticker_E56_N32_BS32_Vanilla_Plasticity_0_005_LR_0_005_L2_5e4_Seed1337",                  SCRIPT_DIR / "sticker" / "variations" / "sticker_e56_n32_bs32_vanilla_plasticity_0_005_lr_0_005_l2_5e4" / "seed_1337" / "predictions.parquet"),
    ("Logistic Regression",            SCRIPT_DIR / "baselines" / "logreg" / "predictions.parquet"),
    ("Kalman Filter",       SCRIPT_DIR / "baselines" / "kalman" / "predictions.parquet"),
    ("ARIMA",            SCRIPT_DIR / "baselines" / "arima" / "predictions.parquet"),
    ("OFI Linear Regression",          SCRIPT_DIR / "baselines" / "ofi_linreg" / "predictions.parquet"),
    ("GARCH(1,1)",             SCRIPT_DIR / "baselines" / "garch" / "predictions.parquet"),
    ("LightGBM",               SCRIPT_DIR / "baselines" / "lgb" / "predictions.parquet"),
    ("DeepLOB",            SCRIPT_DIR / "baselines" / "deeplob" / "predictions.parquet"),
    ("Mamba",                     SCRIPT_DIR / "baselines" / "mamba" / "predictions.parquet"),
    ("TLOB",        SCRIPT_DIR / "baselines" / "tlob" / "predictions.parquet"),
]


def simulate_strategy(preds: np.ndarray, mids: np.ndarray, rt_bps: float) -> dict:
    n   = len(preds)
    cps = rt_bps / 2.0

    m   = mids.astype(np.float64)
    ret = (m[1:] - m[:-1]) / m[:-1] * 10_000.0

    pos      = preds[:-1].astype(np.int8)
    prev_pos = np.empty_like(pos)
    prev_pos[0]  = 0
    prev_pos[1:] = pos[:-1]

    changed = pos != prev_pos

    cost = (
        np.where(changed & (prev_pos != 0), cps, 0.0) +
        np.where(changed & (pos      != 0), cps, 0.0)
    )
    last_pos        = int(pos[-1])
    final_exit_cost = cps if last_pos != 0 else 0.0

    gross_tick      = pos * ret
    net_tick        = gross_tick - cost
    net_tick[-1]   -= final_exit_cost

    cum_gross = np.cumsum(gross_tick)
    cum_net   = np.cumsum(net_tick)

    total_gross_bps = float(cum_gross[-1])
    total_net_bps   = float(cum_net[-1])
    total_cost_bps  = float(np.sum(cost)) + final_exit_cost

    n_ticks      = n - 1
    ticks_per_yr = n_ticks / DATA_DAYS * 365.0

    mu_net = float(np.mean(net_tick))
    sd_net = float(np.std(net_tick))
    net_sharpe_tick = float((mu_net / sd_net) * np.sqrt(ticks_per_yr)) if sd_net > 0 else 0.0

    ann_net_bps = mu_net * ticks_per_yr
    ann_net_pct = ann_net_bps / 10_000.0 * 100.0

    total_net_usd = total_net_bps / 10_000.0 * NOTIONAL_USD

    peak        = np.maximum.accumulate(cum_net)
    dd          = peak - cum_net
    max_dd_bps  = float(np.max(dd))
    peak_val    = float(np.max(cum_net))
    max_dd_pct  = (max_dd_bps / peak_val * 100.0) if peak_val > 0 else 0.0

    calmar = (ann_net_pct / max_dd_pct) if max_dd_pct > 0 else None

    neg = net_tick[net_tick < 0]
    ds  = float(np.std(neg)) if len(neg) > 0 else 1e-10
    net_sortino = float((mu_net / ds) * np.sqrt(ticks_per_yr))

    n_trades            = int(np.sum(changed))
    time_in_market_pct  = float(np.sum(pos != 0)) / n_ticks * 100.0
    signal_chg_rate_pct = n_trades / n_ticks * 100.0
    avg_hold_ticks      = n_ticks / n_trades if n_trades > 0 else float(n_ticks)

    run_id = np.cumsum(changed.astype(np.int32))
    active = pos != 0
    if active.any():
        df_sim = pd.DataFrame({"run_id": run_id[active], "net_pnl": net_tick[active]})
        trade_net = df_sim.groupby("run_id")["net_pnl"].sum().to_numpy()
    else:
        trade_net = np.array([0.0])

    t_wins   = trade_net[trade_net > 0]
    t_losses = trade_net[trade_net <= 0]
    trade_win_rate = len(t_wins) / len(trade_net) * 100.0 if len(trade_net) > 0 else 0.0
    trade_pf = (
        float(np.sum(t_wins)) / abs(float(np.sum(t_losses)))
        if len(t_losses) > 0 and float(np.sum(t_losses)) != 0 else None
    )

    if len(trade_net) > 1 and np.std(trade_net) > 0:
        trades_per_yr = n_trades / DATA_DAYS * 365.0
        trade_sharpe  = float(np.mean(trade_net) / np.std(trade_net) * np.sqrt(trades_per_yr))
    else:
        trade_sharpe = 0.0

    avg_win  = float(np.mean(t_wins))   if len(t_wins)   > 0 else 0.0
    avg_loss = float(np.mean(t_losses)) if len(t_losses) > 0 else 0.0

    breakeven_rt = total_gross_bps / n_trades if n_trades > 0 else None
    in_mkt       = pos != 0
    mu_gross_in  = float(np.mean(gross_tick[in_mkt])) if in_mkt.any() else 0.0
    min_hold_be  = rt_bps / mu_gross_in if mu_gross_in > 0 else None

    return {
        "total_gross_pnl_bps":           round(total_gross_bps, 2),
        "total_cost_bps":                round(total_cost_bps, 2),
        "total_net_pnl_bps":             round(total_net_bps, 2),
        "total_net_pnl_usd":             round(total_net_usd, 2),
        "annualized_net_pct":            round(ann_net_pct, 2),
        "net_sharpe_tick":               round(net_sharpe_tick, 1),
        "net_sharpe_trade":              round(trade_sharpe, 1),
        "net_sortino":                   round(net_sortino, 1),
        "max_drawdown_bps":              round(max_dd_bps, 2),
        "max_drawdown_pct_of_peak":      round(max_dd_pct, 4),
        "calmar_ratio":                  round(calmar, 3) if calmar is not None else None,
        "n_trades":                      n_trades,
        "signal_change_rate_pct":        round(signal_chg_rate_pct, 4),
        "avg_hold_ticks":                round(avg_hold_ticks, 1),
        "time_in_market_pct":            round(time_in_market_pct, 2),
        "trade_win_rate_pct":            round(trade_win_rate, 2),
        "trade_profit_factor":           round(trade_pf, 3) if trade_pf is not None else None,
        "avg_winning_trade_bps":         round(avg_win, 4),
        "avg_losing_trade_bps":          round(avg_loss, 4),
        "breakeven_rt_cost_bps":         round(breakeven_rt, 3) if breakeven_rt is not None else None,
        "min_hold_ticks_for_breakeven":  round(min_hold_be, 1) if min_hold_be is not None else None,
    }


def classify_metrics(preds: np.ndarray, mids: np.ndarray, lat_series) -> dict:
    m     = mids.astype(np.float64)
    diff1 = np.diff(m, append=np.nan)
    act1  = np.where(diff1 > 1e-8, 1, np.where(diff1 < -1e-8, -1, 0))
    valid = ~np.isnan(diff1) & (act1 != 0)
    act_nz, pred_nz = act1[valid], preds[valid]

    da_nz_h1 = float(np.mean(act_nz == pred_nz))
    mcc       = float(matthews_corrcoef(act_nz, pred_nz))
    f1_mac    = float(f1_score(act_nz, pred_nz, average="macro"))
    f1_w      = float(f1_score(act_nz, pred_nz, average="weighted"))
    prec      = float(precision_score(act_nz, pred_nz, average="weighted", zero_division=0))
    rec       = float(recall_score(act_nz, pred_nz, average="weighted", zero_division=0))
    cm        = confusion_matrix(act_nz, pred_nz, labels=[-1, 1]).tolist()

    lat_p50 = float(lat_series.median())        if lat_series is not None else None
    lat_p99 = float(lat_series.quantile(0.99))  if lat_series is not None else None

    ms = pd.Series(m)
    horizon_da = {}
    for h in HORIZONS:
        diff_h = ms.shift(-h).values - m
        act_h  = np.where(diff_h > 1e-8, 1, np.where(diff_h < -1e-8, -1, 0))
        nz_h   = ~np.isnan(diff_h) & (act_h != 0)
        horizon_da[f"h{h}"] = (
            float(np.mean(act_h[nz_h] == preds[nz_h])) if nz_h.sum() > 0 else None
        )

    return {
        "da_nz_h1":         round(da_nz_h1, 4),
        "mcc_h1":           round(mcc, 4),
        "f1_macro_h1":      round(f1_mac, 4),
        "f1_weighted_h1":   round(f1_w, 4),
        "precision_h1":     round(prec, 4),
        "recall_h1":        round(rec, 4),
        "confusion_matrix": cm,
        "horizon_da_nz":    horizon_da,
        "latency_p50_ms":   round(lat_p50, 4) if lat_p50 is not None else None,
        "latency_p99_ms":   round(lat_p99, 4) if lat_p99 is not None else None,
    }


def compute_model_metrics(name: str, pred_path: Path) -> dict | None:
    if not pred_path.exists():
        print(f"[SKIP] {name}: file not found.", flush=True)
        return None

    t0 = time.time()
    print(f"[PROCESS] {name} ...", flush=True)

    df = pd.read_parquet(pred_path).sort_values("tick_idx").reset_index(drop=True)
    df["mid"]  = df["mid"].astype(np.float64)
    df["pred"] = df["pred"].astype(np.int8)

    total_ticks = len(df)
    n_75 = total_ticks // 4
    df = df.iloc[n_75:].reset_index(drop=True)

    preds = df["pred"].to_numpy()
    mids  = df["mid"].to_numpy()
    lat   = df["lat_ms"] if "lat_ms" in df.columns else None

    cls = classify_metrics(preds, mids, lat)

    scenarios = {}
    for sc_name, rt_bps in COST_SCENARIOS.items():
        scenarios[sc_name] = simulate_strategy(preds, mids, rt_bps)

    elapsed  = time.time() - t0
    real_sc  = scenarios["realistic"]
    print(
        f"[DONE]  {name} | {elapsed:.1f}s | "
        f"DA={cls['da_nz_h1']:.4f} | "
        f"Net(7bps)={real_sc['total_net_pnl_bps']:+.0f}bps | "
        f"Trades={real_sc['n_trades']:,} | "
        f"AvgHold={real_sc['avg_hold_ticks']:.1f}tk",
        flush=True,
    )

    return {"model_name": name, "eval_ticks": len(df), **cls, "scenarios": scenarios}


def generate_report(results: list) -> None:
    rows = []
    for r in results:
        base = {k: v for k, v in r.items() if k not in ("scenarios", "horizon_da_nz", "confusion_matrix")}
        for sc_name, sc in r.get("scenarios", {}).items():
            for k, v in sc.items():
                base[f"{sc_name}__{k}"] = v
        rows.append(base)
    df_out = pd.DataFrame(rows).sort_values("da_nz_h1", ascending=False).reset_index(drop=True)
    df_out.to_csv(OUT_DIR / "test_rank.csv", index=False)

    with open(OUT_DIR / "test_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    sc_names = list(COST_SCENARIOS.keys())
    sorted_net = sorted(results, key=lambda x: -x["scenarios"]["realistic"]["total_net_pnl_bps"])

    md = [
        "# Realistic Trading Profitability Leaderboard",
        "",
        "**Dataset**: 11,918,929 Hyperliquid BTC Perpetual Futures trade ticks (Dec 2025)  ",
        "**Evaluation Basis**: Last 75% of ticks  ",
        "**Protocol**: Zero-lookahead prequential (test-then-train) streaming  ",
        "**Strategy**: Signal-following - hold position until signal reversal  ",
        "**Notional**: $100,000 USD fixed per position (scales linearly)  ",
        "",
        "---",
        "",
        "## Cost Model (Hyperliquid BTC Perp)",
        "",
        "| Component | Per Side | Round-Trip |",
        "|:---|:---:|:---:|",
        "| Taker fee (standard user) | 2.5 bps | 5.0 bps |",
        "| Half bid-ask spread (conservative) | 0.5 bps | 1.0 bps |",
        "| Latency / queue slippage | 0.5 bps | 1.0 bps |",
        "| **Total base case** | **3.5 bps** | **7.0 bps** |",
        "",
        "---",
        "",
        "## 1. Classification Metrics (h=1, Last 75% Ticks)",
        "",
        "| Rank | Model | DA (h=1) | MCC | Weighted F1 | Latency p99 (ms) |",
        "|:---:|:---|:---:|:---:|:---:|:---:|",
    ]
    for i, r in enumerate(sorted(results, key=lambda x: -x["da_nz_h1"]), 1):
        lat = f"{r['latency_p99_ms']:.2f}" if r.get("latency_p99_ms") is not None else "N/A"
        md.append(f"| {i} | **{r['model_name']}** | **{r['da_nz_h1']:.4f}** | {r['mcc_h1']:.4f} | {r['f1_weighted_h1']:.4f} | {lat} |")

    md += [
        "", "---", "",
        "## 2. Net Profitability After Realistic Costs (7.0 bps round-trip)",
        "",
        "| Rank | Model | Gross (bps) | Cost (bps) | **Net (bps)** | Net (USD) | Ann. Net % | Net Sharpe (tick) | Net Sharpe (trade) | Max DD % | Calmar |",
        "|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for i, r in enumerate(sorted_net, 1):
        sc = r["scenarios"]["realistic"]
        calmar = f"{sc['calmar_ratio']:.2f}" if sc.get("calmar_ratio") is not None else "N/A"
        md.append(
            f"| {i} | **{r['model_name']}** "
            f"| {sc['total_gross_pnl_bps']:+.0f} "
            f"| {sc['total_cost_bps']:.0f} "
            f"| **{sc['total_net_pnl_bps']:+.0f}** "
            f"| ${sc['total_net_pnl_usd']:+,.0f} "
            f"| {sc['annualized_net_pct']:+.1f}% "
            f"| {sc['net_sharpe_tick']:.1f} "
            f"| {sc['net_sharpe_trade']:.1f} "
            f"| {sc['max_drawdown_pct_of_peak']:.2f}% "
            f"| {calmar} |"
        )

    md += [
        "", "---", "",
        "## 3. Turnover & Breakeven Analysis",
        "",
        "| Model | # Trades | Chg Rate | Avg Hold | In Mkt | Trade Win Rate | Trade PF | BE RT (bps) | Min Hold |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r in sorted_net:
        sc  = r["scenarios"]["realistic"]
        tpf = f"{sc['trade_profit_factor']:.3f}" if sc.get("trade_profit_factor") is not None else "N/A"
        be  = f"{sc['breakeven_rt_cost_bps']:.2f}" if sc.get("breakeven_rt_cost_bps") is not None else "N/A"
        mh  = f"{sc['min_hold_ticks_for_breakeven']:.0f}" if sc.get("min_hold_ticks_for_breakeven") is not None else "N/A"
        md.append(
            f"| **{r['model_name']}** "
            f"| {sc['n_trades']:,} "
            f"| {sc['signal_change_rate_pct']:.2f}% "
            f"| {sc['avg_hold_ticks']:.1f} "
            f"| {sc['time_in_market_pct']:.1f}% "
            f"| {sc['trade_win_rate_pct']:.1f}% "
            f"| {tpf} "
            f"| {be} "
            f"| {mh} |"
        )

    md += [
        "", "---", "",
        "## 4. Net P&L Sensitivity to Transaction Cost (bps)",
        "",
        "| Model | Zero-Cost | Optimistic (4 bps RT) | Realistic (7 bps RT) | Conservative (12 bps RT) | Pessimistic (18 bps RT) |",
        "|:---|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r in sorted_net:
        row = f"| **{r['model_name']}** "
        for sc_name in sc_names:
            v = r["scenarios"][sc_name]["total_net_pnl_bps"]
            row += f"| {v:+.0f} "
        row += "|"
        md.append(row)

    md += [
        "", "---", "",
        "## 5. Multi-Horizon Non-Zero Directional Accuracy (Last 75% Ticks, first 12 models)",
        "",
        "| h | " + " | ".join(r["model_name"][:25] for r in results[:12]) + " |",
        "|:---:|" + ":---:|" * min(12, len(results)),
    ]
    for h in HORIZONS:
        row = f"| **h={h}** "
        for r in results[:12]:
            v = r.get("horizon_da_nz", {}).get(f"h{h}")
            row += f"| {v:.4f} " if v is not None else "| N/A "
        row += "|"
        md.append(row)

    md += [
        "", "---", "",
        "## Methodology Notes",
        "",
        "1. **Evaluation basis**: Last 75% of ticks by row count (first 25% discarded).",
        "2. **Signal-following**: `position[t] = pred[t]`. Trade when `pred[t] != pred[t-1]`.",
        "3. **Cost**: entry (rt/2) at open, exit (rt/2) at close. Reversal = 1 full round-trip.",
        "4. **Sharpe (tick)**: `mean(net)/std(net) x sqrt(n_ticks/31 x 365)`.",
        "5. **Sharpe (trade)**: `mean(trade_net)/std(trade_net) x sqrt(n_trades/31 x 365)`.",
        "6. **Max DD**: drawdown in bps from peak cumulative net equity, expressed as % of that peak.",
        "7. **Calmar**: annualized net return % / max DD %.",
        "8. **Notional**: $100,000 USD. Dollar P&L scales linearly with position size.",
        "9. Funding rate, mark-to-market margin, and borrow costs are NOT included.",
    ]

    rpt = OUT_DIR / "README.md"
    with open(rpt, "w") as f:
        f.write("\n".join(md) + "\n")

    print(f"\n[SAVED] {OUT_DIR / 'test_metrics.json'}")
    print(f"[SAVED] {OUT_DIR / 'test_rank.csv'}")
    print(f"[SAVED] {rpt}")


def main() -> None:
    print("=" * 85, flush=True)
    print("  REALISTIC TRADING PROFITABILITY - LAST 75% TICKS EVALUATION")
    print(f"  Cost model : {COST_PER_SIDE} bps/side  |  {DEFAULT_RT_BPS} bps round-trip (realistic)")
    print(f"  Notional   : ${NOTIONAL_USD:,} USD fixed per position")
    print("=" * 85, flush=True)

    results = []
    for name, path in MODEL_CONFIGS:
        m = compute_model_metrics(name, path)
        if m is not None:
            results.append(m)

    generate_report(results)

    print("\n" + "=" * 85, flush=True)
    print("  COMPLETE!")
    print(f"  Report: {OUT_DIR / 'README.md'}")
    print("=" * 85, flush=True)


if __name__ == "__main__":
    main()
