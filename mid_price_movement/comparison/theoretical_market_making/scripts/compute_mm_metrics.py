import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR    = Path(__file__).resolve().parent.parent.parent.parent
OUT_DIR     = Path(__file__).resolve().parent.parent
MASTER_PATH = BASE_DIR / "data" / "extracted" / "btc_trades_master.parquet"
RESULTS_DIR = OUT_DIR / "sticker_e56_n56_bs128_default"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HALF_SPREAD_BPS  = 0.5
MAKER_REBATE_BPS = 1.0
NOTIONAL_USD     = 100_000
VPIN_BUCKET_SIZE = 200
ADV_SEL_HORIZONS = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500]
HORIZONS         = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500]
DATA_DAYS        = 31.0
LOGIT_THRESHOLD  = 0.0

MODEL_CONFIGS = [
    ("Sticker",                 BASE_DIR / "sticker" / "variations" / "sticker_e56_n56_bs128_default" / "predictions.parquet"),
    ("Sticker_E56_N32_BS128_Vanilla_Plasticity_0_005_LR_0_001_L2_1e3",  BASE_DIR / "sticker" / "variations" / "sticker_e56_n32_bs128_vanilla_plasticity_0_005_lr_0_001_l2_1e3" / "predictions.parquet"),
    ("Sticker_E56_N40_BS128_Vanilla_Plasticity_0_005_LR_0_005_L2_5e4",         BASE_DIR / "sticker" / "variations" / "sticker_e56_n40_bs128_vanilla_plasticity_0_005_lr_0_005_l2_5e4" / "predictions.parquet"),
    ("Sticker_E56_N48_BS128_Vanilla_Plasticity_0_005_LR_0_005_L2_5e4",         BASE_DIR / "sticker" / "variations" / "sticker_e56_n48_bs128_vanilla_plasticity_0_005_lr_0_005_l2_5e4" / "predictions.parquet"),
    ("Sticker_E56_N48_BS128_Vanilla_Plasticity_0_005_LR_0_001_L2_1e3",  BASE_DIR / "sticker" / "variations" / "sticker_e56_n48_bs128_vanilla_plasticity_0_005_lr_0_001_l2_1e3" / "predictions.parquet"),
    ("Sticker_E56_N32_BS32_FastAdapt_Plasticity_0_02_LR_0_003_L2_1e5", BASE_DIR / "sticker" / "variations" / "sticker_e56_n32_bs32_fastadapt_plasticity_0_02_lr_0_003_l2_1e5" / "predictions.parquet"),
    ("Sticker_E56_N32_BS32_Vanilla_Plasticity_0_005_LR_0_005_L2_5e4_Seed42",    BASE_DIR / "sticker" / "variations" / "sticker_e56_n32_bs32_vanilla_plasticity_0_005_lr_0_005_l2_5e4" / "default" / "predictions.parquet"),
    ("Sticker_E16_N16_BS128_Vanilla_Plasticity_0_0005_LR_0_0005_L2_1e4",         BASE_DIR / "sticker" / "variations" / "sticker_e16_n16_bs128_vanilla_plasticity_0_0005_lr_0_0005_l2_1e4" / "predictions.parquet"),
    ("Sticker_E28_N32_BS128_Vanilla_Plasticity_0_005_LR_0_005_L2_5e4",         BASE_DIR / "sticker" / "variations" / "sticker_e28_n32_bs128_vanilla_plasticity_0_005_lr_0_005_l2_5e4" / "predictions.parquet"),
    ("Sticker_E56_N56_BS128_Vanilla_Plasticity_0_005_LR_0_005_L2_5e4",         BASE_DIR / "sticker" / "variations" / "sticker_e56_n56_bs128_vanilla_plasticity_0_005_lr_0_005_l2_5e4" / "predictions.parquet"),
    ("Sticker_E56_N64_BS128_Vanilla_Plasticity_0_005_LR_0_005_L2_5e4",         BASE_DIR / "sticker" / "variations" / "sticker_e56_n64_bs128_vanilla_plasticity_0_005_lr_0_005_l2_5e4" / "predictions.parquet"),
    ("Sticker_E56_N112_BS128_Vanilla_Plasticity_0_005_LR_0_005_L2_5e4",        BASE_DIR / "sticker" / "variations" / "sticker_e56_n112_bs128_vanilla_plasticity_0_005_lr_0_005_l2_5e4" / "predictions.parquet"),
    ("Sticker_E112_N32_BS128_Vanilla_Plasticity_0_005_LR_0_005_L2_5e4",        BASE_DIR / "sticker" / "variations" / "sticker_e112_n32_bs128_vanilla_plasticity_0_005_lr_0_005_l2_5e4" / "predictions.parquet"),
    ("Sticker_E112_N64_BS128_Vanilla_Plasticity_0_005_LR_0_005_L2_5e4",        BASE_DIR / "sticker" / "variations" / "sticker_e112_n64_bs128_vanilla_plasticity_0_005_lr_0_005_l2_5e4" / "predictions.parquet"),
    ("Sticker_E56_N32_BS256_Vanilla_Plasticity_0_005_LR_0_001_L2_5e4",   BASE_DIR / "sticker" / "variations" / "sticker_e56_n32_bs256_vanilla_plasticity_0_005_lr_0_001_l2_5e4" / "predictions.parquet"),
    ("Sticker_E56_N56_BS256_Vanilla_Plasticity_0_005_LR_0_001_L2_5e4",   BASE_DIR / "sticker" / "variations" / "sticker_e56_n56_bs256_vanilla_plasticity_0_005_lr_0_001_l2_5e4" / "predictions.parquet"),
    ("Sticker_E56_N32_BS256_Orthogonal_Plasticity_0_005_LR_0_001_L2_5e4", BASE_DIR / "sticker" / "variations" / "sticker_e56_n32_bs256_orthogonal_plasticity_0_005_lr_0_001_l2_5e4" / "predictions.parquet"),
    ("Sticker_E56_N32_BS128_Orthogonal_L2_0", BASE_DIR / "sticker" / "sticker_e56_n32_bs128_orthogonal_l2_0" / "predictions.parquet"),
    ("Sticker_E56_N32_BS128_Orthogonal_Plasticity_0_01_L2_0", BASE_DIR / "sticker" / "sticker_e56_n32_bs128_orthogonal_plasticity_0_01_l2_0" / "default" / "predictions.parquet"),
    ("Sticker_E56_N32_BS128_Orthogonal_Plasticity_0_1_L2_0", BASE_DIR / "sticker" / "sticker_e56_n32_bs128_orthogonal_plasticity_0_1_l2_0" / "predictions.parquet"),
    ("Sticker_E56_N32_BS256_Orthogonal_L2_0", BASE_DIR / "sticker" / "sticker_e56_n32_bs256_orthogonal_l2_0" / "predictions.parquet"),
    ("Sticker_E56_N32_BS256_Orthogonal_Plasticity_0_01_L2_0", BASE_DIR / "sticker" / "sticker_e56_n32_bs256_orthogonal_plasticity_0_01_l2_0" / "default" / "predictions.parquet"),
    ("Sticker_E56_N32_BS256_Orthogonal_Plasticity_0_01_L2_0_Seed67", BASE_DIR / "sticker" / "sticker_e56_n32_bs256_orthogonal_plasticity_0_01_l2_0" / "seed_67" / "predictions.parquet"),
    ("Sticker_E56_N32_BS256_Orthogonal_Plasticity_0_01_L2_0_Seed420", BASE_DIR / "sticker" / "sticker_e56_n32_bs256_orthogonal_plasticity_0_01_l2_0" / "seed_420" / "predictions.parquet"),
    ("Sticker_E56_N32_BS256_Orthogonal_Plasticity_0_1_L2_0", BASE_DIR / "sticker" / "sticker_e56_n32_bs256_orthogonal_plasticity_0_1_l2_0" / "predictions.parquet"),
    ("Sticker_E56_N56_BS128_Orthogonal_Plasticity_0_01_L2_0", BASE_DIR / "sticker" / "sticker_e56_n56_bs128_orthogonal_plasticity_0_01_l2_0" / "default" / "predictions.parquet"),
    ("Sticker_E56_N56_BS256_Orthogonal_Plasticity_0_01_L2_0", BASE_DIR / "sticker" / "sticker_e56_n56_bs256_orthogonal_plasticity_0_01_l2_0" / "predictions.parquet"),
    ("Sticker_E56_N32_BS256_Orthogonal_Plasticity_0_01_LR_0_003_L2_0", BASE_DIR / "sticker" / "sticker_e56_n32_bs256_orthogonal_plasticity_0_01_lr_0_003_l2_0" / "predictions.parquet"),
    ("Sticker_E56_N56_BS256_Orthogonal_Plasticity_0_01_LR_0_003_L2_0", BASE_DIR / "sticker" / "sticker_e56_n56_bs256_orthogonal_plasticity_0_01_lr_0_003_l2_0" / "predictions.parquet"),
    ("Sticker_E56_N32_BS128_Orthogonal_Plasticity_0_01_LR_0_001_L2_1e6", BASE_DIR / "sticker" / "variations" / "sticker_e56_n32_bs128_orthogonal_plasticity_0_01_lr_0_001_l2_1e6" / "predictions.parquet"),
    ("Sticker_E56_N32_BS256_Orthogonal_Plasticity_0_01_LR_0_001_L2_1e6", BASE_DIR / "sticker" / "variations" / "sticker_e56_n32_bs256_orthogonal_plasticity_0_01_lr_0_001_l2_1e6" / "predictions.parquet"),
    ("Sticker_E56_N48_BS256_Orthogonal_Plasticity_0_01_LR_0_001_L2_0", BASE_DIR / "sticker" / "variations" / "sticker_e56_n48_bs256_orthogonal_plasticity_0_01_lr_0_001_l2_0" / "predictions.parquet"),
    ("Sticker_E56_N32_BS256_Orthogonal_Plasticity_0_01_LR_0_001_L2_5e6", BASE_DIR / "sticker" / "variations" / "sticker_e56_n32_bs256_orthogonal_plasticity_0_01_lr_0_001_l2_5e6" / "predictions.parquet"),
    ("Sticker_E56_N32_BS256_Orthogonal_Plasticity_0_01_LR_0_001_L2_1e5", BASE_DIR / "sticker" / "variations" / "sticker_e56_n32_bs256_orthogonal_plasticity_0_01_lr_0_001_l2_1e5" / "predictions.parquet"),
    ("Sticker_E56_N32_BS256_Orthogonal_Plasticity_0_01_LR_0_001_L2_5e5", BASE_DIR / "sticker" / "variations" / "sticker_e56_n32_bs256_orthogonal_plasticity_0_01_lr_0_001_l2_5e5" / "predictions.parquet"),
    ("Sticker_E56_N32_BS256_Orthogonal_Plasticity_0_1_LR_0_001_L2_1e6", BASE_DIR / "sticker" / "variations" / "sticker_e56_n32_bs256_orthogonal_plasticity_0_1_lr_0_001_l2_1e6" / "predictions.parquet"),
    ("Sticker_E56_N32_BS256_Orthogonal_Plasticity_0_01_LR_0_0005_L2_0", BASE_DIR / "sticker" / "sticker_e56_n32_bs256_orthogonal_plasticity_0_01_lr_0_0005_l2_0" / "predictions.parquet"),
    ("Sticker_E56_N56_BS256_Orthogonal_Plasticity_0_01_LR_0_0005_L2_0", BASE_DIR / "sticker" / "sticker_e56_n56_bs256_orthogonal_plasticity_0_01_lr_0_0005_l2_0" / "default" / "predictions.parquet"),
    ("Sticker_E56_N56_BS256_Orthogonal_Plasticity_0_01_LR_0_0005_L2_0_Seed67", BASE_DIR / "sticker" / "sticker_e56_n56_bs256_orthogonal_plasticity_0_01_lr_0_0005_l2_0" / "seed_67" / "predictions.parquet"),
    ("Sticker_E56_N56_BS256_Orthogonal_Plasticity_0_01_LR_0_0005_L2_0_Seed420", BASE_DIR / "sticker" / "sticker_e56_n56_bs256_orthogonal_plasticity_0_01_lr_0_0005_l2_0" / "seed_420" / "predictions.parquet"),
    ("Sticker_E56_N56_BS128_Orthogonal_Plasticity_0_01_L2_0_Seed67", BASE_DIR / "sticker" / "sticker_e56_n56_bs128_orthogonal_plasticity_0_01_l2_0" / "seed_67" / "predictions.parquet"),
    ("Sticker_E56_N56_BS128_Orthogonal_Plasticity_0_01_L2_0_Seed420", BASE_DIR / "sticker" / "sticker_e56_n56_bs128_orthogonal_plasticity_0_01_l2_0" / "seed_420" / "predictions.parquet"),
    ("Sticker_E56_N32_BS128_Orthogonal_Plasticity_0_01_L2_0_Seed67", BASE_DIR / "sticker" / "sticker_e56_n32_bs128_orthogonal_plasticity_0_01_l2_0" / "seed_67" / "predictions.parquet"),
    ("Sticker_E56_N32_BS128_Orthogonal_Plasticity_0_01_L2_0_Seed420", BASE_DIR / "sticker" / "sticker_e56_n32_bs128_orthogonal_plasticity_0_01_l2_0" / "seed_420" / "predictions.parquet"),
    ("Sticker_E56_N56_BS128_Orthogonal_Plasticity_0_1_L2_0", BASE_DIR / "sticker" / "sticker_e56_n56_bs128_orthogonal_plasticity_0_1_l2_0" / "predictions.parquet"),
    ("Sticker_E56_N56_BS128_Orthogonal_Plasticity_0_05_L2_0", BASE_DIR / "sticker" / "sticker_e56_n56_bs128_orthogonal_plasticity_0_05_l2_0" / "predictions.parquet"),
    ("Sticker_E56_N32_BS128_Orthogonal_Plasticity_0_01_LR_0_0005_L2_0", BASE_DIR / "sticker" / "sticker_e56_n32_bs128_orthogonal_plasticity_0_01_lr_0_0005_l2_0" / "default" / "predictions.parquet"),
    ("Sticker_E56_N32_BS128_Orthogonal_Plasticity_0_01_LR_0_0005_L2_0_Seed67", BASE_DIR / "sticker" / "sticker_e56_n32_bs128_orthogonal_plasticity_0_01_lr_0_0005_l2_0" / "seed_67" / "predictions.parquet"),
    ("Sticker_E56_N32_BS128_Orthogonal_Plasticity_0_01_LR_0_0005_L2_0_Seed420", BASE_DIR / "sticker" / "sticker_e56_n32_bs128_orthogonal_plasticity_0_01_lr_0_0005_l2_0" / "seed_420" / "predictions.parquet"),
    ("Sticker_E56_N56_BS128_Orthogonal_Plasticity_0_01_L2_0_Seed1", BASE_DIR / "sticker" / "sticker_e56_n56_bs128_orthogonal_plasticity_0_01_l2_0" / "seed_1" / "predictions.parquet"),
    ("Sticker_E56_N56_BS128_Orthogonal_Plasticity_0_01_L2_0_Seed999", BASE_DIR / "sticker" / "sticker_e56_n56_bs128_orthogonal_plasticity_0_01_l2_0" / "seed_999" / "predictions.parquet"),
    ("Sticker_E56_N32_BS128_Orthogonal_Plasticity_0_01_L2_0_Seed999", BASE_DIR / "sticker" / "sticker_e56_n32_bs128_orthogonal_plasticity_0_01_l2_0" / "seed_999" / "predictions.parquet"),
    ("Sticker_E56_N56_BS256_Orthogonal_Plasticity_0_01_LR_0_0005_L2_0_Seed1", BASE_DIR / "sticker" / "sticker_e56_n56_bs256_orthogonal_plasticity_0_01_lr_0_0005_l2_0" / "seed_1" / "predictions.parquet"),
    ("Sticker_E56_N32_BS256_Orthogonal_Plasticity_0_01_L2_0_Seed1", BASE_DIR / "sticker" / "sticker_e56_n32_bs256_orthogonal_plasticity_0_01_l2_0" / "seed_1" / "predictions.parquet"),
    ("Sticker_E56_N56_BS128_Orthogonal_Plasticity_0_01_LR_0_0005_L2_0", BASE_DIR / "sticker" / "sticker_e56_n56_bs128_orthogonal_plasticity_0_01_lr_0_0005_l2_0" / "predictions.parquet"),
    ("Sticker_E56_N56_BS256_Orthogonal_Plasticity_0_01_LR_0_00001_L2_0", BASE_DIR / "sticker" / "variations" / "sticker_e56_n56_bs256_orthogonal_plasticity_0_01_lr_0_00001_l2_0" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_Plasticity_0_01_L2_0_Seed67", BASE_DIR / "sticker" / "sticker_e56_n32_orthogonal_plasticity_0_01_l2_0" / "seed_67" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_Plasticity_0_01_L2_0_Seed420", BASE_DIR / "sticker" / "sticker_e56_n32_orthogonal_plasticity_0_01_l2_0" / "seed_420" / "predictions.parquet"),
    ("Sticker_E56_N56_BS256_Orthogonal_Plasticity_0_005_LR_0_001_L2_5e4",   BASE_DIR / "sticker" / "variations" / "sticker_e56_n56_bs256_orthogonal_plasticity_0_005_lr_0_001_l2_5e4" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_Plasticity_0_005_LR_0_001_L2_5e4",     BASE_DIR / "sticker" / "variations" / "sticker_e56_n32_orthogonal_plasticity_0_005_lr_0_001_l2_5e4" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_Plasticity_0_005_LR_0_001_L2_1e3", BASE_DIR / "sticker" / "variations" / "sticker_e56_n32_orthogonal_plasticity_0_005_lr_0_001_l2_1e3" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_Plasticity_0_005_LR_0_001_L2_1e4", BASE_DIR / "sticker" / "variations" / "sticker_e56_n32_orthogonal_plasticity_0_005_lr_0_001_l2_1e4" / "predictions.parquet"),
    ("Sticker_E56_N48_Orthogonal_Plasticity_0_005_LR_0_001_L2_5e4",     BASE_DIR / "sticker" / "variations" / "sticker_e56_n48_orthogonal_plasticity_0_005_lr_0_001_l2_5e4" / "predictions.parquet"),
    ("Sticker_E56_N56_Orthogonal_Plasticity_0_005_LR_0_001_L2_5e4",     BASE_DIR / "sticker" / "variations" / "sticker_e56_n56_orthogonal_plasticity_0_005_lr_0_001_l2_5e4" / "predictions.parquet"),
    ("Sticker_E56_N56_Orthogonal_Plasticity_0_005_LR_0_001_L2_1e4", BASE_DIR / "sticker" / "variations" / "sticker_e56_n56_orthogonal_plasticity_0_005_lr_0_001_l2_1e4" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_Plasticity_1e3_LR_0_001_L2_5e4", BASE_DIR / "sticker" / "variations" / "sticker_e56_n32_orthogonal_plasticity_1e3_lr_0_001_l2_5e4" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_Plasticity_0_005_LR_0_001_L2_1e5", BASE_DIR / "sticker" / "variations" / "sticker_e56_n32_orthogonal_plasticity_0_005_lr_0_001_l2_1e5" / "predictions.parquet"),
    ("Sticker_E56_N56_Orthogonal_Plasticity_0_005_LR_0_001_L2_1e5", BASE_DIR / "sticker" / "variations" / "sticker_e56_n56_orthogonal_plasticity_0_005_lr_0_001_l2_1e5" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_Plasticity_0_005_LR_0_001_L2_1e6", BASE_DIR / "sticker" / "variations" / "sticker_e56_n32_orthogonal_plasticity_0_005_lr_0_001_l2_1e6" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_Plasticity_0_005_LR_0_001_L2_1e7", BASE_DIR / "sticker" / "variations" / "sticker_e56_n32_orthogonal_plasticity_0_005_lr_0_001_l2_1e7" / "predictions.parquet"),
    ("Sticker_E56_N56_Orthogonal_Plasticity_0_005_LR_0_001_L2_1e6", BASE_DIR / "sticker" / "variations" / "sticker_e56_n56_orthogonal_plasticity_0_005_lr_0_001_l2_1e6" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_Plasticity_0_005_LR_0_001_L2_1e10", BASE_DIR / "sticker" / "variations" / "sticker_e56_n32_orthogonal_plasticity_0_005_lr_0_001_l2_1e10" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_L2_0",    BASE_DIR / "sticker" / "sticker_e56_n32_orthogonal_l2_0" / "default" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_Plasticity_0_01_L2_0", BASE_DIR / "sticker" / "sticker_e56_n32_orthogonal_plasticity_0_01_l2_0" / "default" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_Plasticity_0_05_LR_0_001_L2_0", BASE_DIR / "sticker" / "sticker_e56_n32_orthogonal_plasticity_0_05_lr_0_001_l2_0" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_Plasticity_0_1_LR_0_001_L2_0",  BASE_DIR / "sticker" / "sticker_e56_n32_orthogonal_plasticity_0_1_lr_0_001_l2_0" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_L2_0_Seed1337",   BASE_DIR / "sticker" / "sticker_e56_n32_orthogonal_l2_0" / "seed_1337" / "predictions.parquet"),
    ("Sticker_E56_N56_Orthogonal_Plasticity_0_005_LR_0_001_L2_0",            BASE_DIR / "sticker" / "variations" / "sticker_e56_n56_orthogonal_plasticity_0_005_lr_0_001_l2_0" / "predictions.parquet"),
    ("Sticker_E56_N32_Orthogonal_PurePlasticity_LR_0_001_L2_0",   BASE_DIR / "sticker" / "variations" / "sticker_e56_n32_orthogonal_pure_plasticity_lr_0_001_l2_0" / "predictions.parquet"),
    ("Sticker_E56_N32_BS32_Vanilla_Plasticity_0_005_LR_0_005_L2_5e4_Seed1337",                  BASE_DIR / "sticker" / "variations" / "sticker_e56_n32_bs32_vanilla_plasticity_0_005_lr_0_005_l2_5e4" / "seed_1337" / "predictions.parquet"),
    ("Logistic Regression",            BASE_DIR / "baselines" / "logreg" / "predictions.parquet"),
    ("Kalman Filter",       BASE_DIR / "baselines" / "kalman" / "predictions.parquet"),
    ("ARIMA",            BASE_DIR / "baselines" / "arima" / "predictions.parquet"),
    ("OFI Linear Regression",          BASE_DIR / "baselines" / "ofi_linreg" / "predictions.parquet"),
    ("GARCH(1,1)",             BASE_DIR / "baselines" / "garch" / "predictions.parquet"),
    ("LightGBM",               BASE_DIR / "baselines" / "lgb" / "predictions.parquet"),
    ("DeepLOB",            BASE_DIR / "baselines" / "deeplob" / "predictions.parquet"),
    ("Mamba",                     BASE_DIR / "baselines" / "mamba" / "predictions.parquet"),
    ("TLOB",        BASE_DIR / "baselines" / "tlob" / "predictions.parquet"),
]


def compute_vpin(side_sign: np.ndarray, notional: np.ndarray, bucket_size: int = VPIN_BUCKET_SIZE) -> tuple[float, float]:
    n = len(side_sign)
    n_buckets = n // bucket_size
    if n_buckets < 2:
        return 0.0, 0.0
    vpins = []
    for i in range(n_buckets):
        sl = slice(i * bucket_size, (i + 1) * bucket_size)
        v_buy  = float(np.sum(notional[sl][side_sign[sl] == 1]))
        v_sell = float(np.sum(notional[sl][side_sign[sl] == -1]))
        v_tot  = v_buy + v_sell
        if v_tot > 0:
            vpins.append(abs(v_buy - v_sell) / v_tot)
    if not vpins:
        return 0.0, 0.0
    return float(np.mean(vpins)), float(np.std(vpins))


def inventory_half_life(inventory: np.ndarray) -> float:
    if len(inventory) < 10:
        return -1.0
    y = inventory[1:]
    x = inventory[:-1]
    if np.std(x) < 1e-12:
        return -1.0
    phi = float(np.cov(x, y)[0, 1] / np.var(x))
    if phi <= 0 or phi >= 1:
        return -1.0
    return float(-np.log(2) / np.log(phi))


def sharpe(pnl_ticks: np.ndarray, ticks_per_yr: float) -> float:
    mu, sd = float(np.mean(pnl_ticks)), float(np.std(pnl_ticks))
    return float((mu / sd) * np.sqrt(ticks_per_yr)) if sd > 0 else 0.0


def sortino(pnl_ticks: np.ndarray, ticks_per_yr: float) -> float:
    mu = float(np.mean(pnl_ticks))
    neg = pnl_ticks[pnl_ticks < 0]
    ds  = float(np.std(neg)) if len(neg) > 0 else 1e-10
    return float((mu / ds) * np.sqrt(ticks_per_yr))


def compute_metrics(name: str, pred_path: Path, master: pd.DataFrame) -> dict | None:
    if not pred_path.exists():
        print(f"[SKIP] {name}", flush=True)
        return None

    t0 = time.time()
    print(f"[PROCESS] {name} ...", flush=True)

    import pyarrow.parquet as _pq
    _schema_names = _pq.read_schema(pred_path).names
    _cols = ["tick_idx", "mid", "pred"] + (["logit"] if "logit" in _schema_names else [])
    preds = pd.read_parquet(pred_path, columns=_cols).rename(columns={"mid": "mid_pred"})
    if "logit" not in preds.columns:
        preds["logit"] = 1.0
    df = master.merge(preds, on="tick_idx", how="inner").reset_index(drop=True)
    n_total = len(df)
    if n_total < 100:
        print(f"[SKIP] {name}: too few joined rows ({n_total})", flush=True)
        return None

    n_75 = n_total // 4
    df = df.iloc[n_75:].reset_index(drop=True)
    n  = len(df)

    ticks_per_yr = n / DATA_DAYS * 365.0

    px        = df["px"].to_numpy(dtype=np.float64)
    sz        = df["sz"].to_numpy(dtype=np.float64)
    notional  = df["notional"].to_numpy(dtype=np.float64)
    side_sign = df["side_sign"].to_numpy(dtype=np.int8)
    pred      = df["pred"].to_numpy(dtype=np.int8)
    logit     = df["logit"].to_numpy(dtype=np.float64)
    mid       = df["mid_pred"].to_numpy(dtype=np.float64)

    diff_mid1 = np.diff(mid, append=np.nan)
    act_dir1  = np.where(diff_mid1 > 1e-8, 1, np.where(diff_mid1 < -1e-8, -1, 0))
    nz_dir    = ~np.isnan(diff_mid1) & (act_dir1 != 0)
    da_nz_h1  = float(np.mean(act_dir1[nz_dir] == pred[nz_dir])) if nz_dir.sum() > 0 else 0.5

    horizon_da_nz = {}
    ms_mid = pd.Series(mid)
    for h in HORIZONS:
        diff_h = ms_mid.shift(-h).values - mid
        act_h  = np.where(diff_h > 1e-8, 1, np.where(diff_h < -1e-8, -1, 0))
        nz_h   = ~np.isnan(diff_h) & (act_h != 0)
        horizon_da_nz[f"h{h}"] = float(np.mean(act_h[nz_h] == pred[nz_h])) if nz_h.sum() > 0 else None

    eff_hs_bps = np.abs(px - mid) / mid * 10_000.0

    price_impact_k, realized_hs_k = {}, {}
    for k in [5, 20]:
        mid_shift = np.roll(mid, -k)
        mid_shift[-k:] = np.nan
        dm = (mid_shift - mid) / mid * 10_000.0
        pi = side_sign * dm
        rs = HALF_SPREAD_BPS - pi
        valid = ~np.isnan(dm)
        price_impact_k[k]  = float(np.nanmean(pi))
        realized_hs_k[k]   = float(np.nanmean(rs))

    adv_sel, adv_sel_rate = {}, {}
    for k in ADV_SEL_HORIZONS:
        mid_shift = np.roll(mid, -k)
        mid_shift[-k:] = np.nan
        dm    = (mid_shift - mid) / mid * 10_000.0
        inv_sign = -side_sign.astype(np.float64)
        adv   = inv_sign * dm
        valid = ~np.isnan(adv)
        adv_sel[k]      = float(np.nanmean(adv[valid]))
        adv_sel_rate[k] = float(np.nanmean(adv[valid] > 0)) * 100.0

    mid_shift5 = np.roll(mid, -5); mid_shift5[-5:] = np.nan
    Deltamid5 = (mid_shift5 - mid) / mid * 10_000.0
    signed_vol = side_sign * notional
    valid5 = ~np.isnan(Deltamid5)
    if valid5.sum() > 0 and np.std(signed_vol[valid5]) > 0:
        cov = np.cov(signed_vol[valid5], Deltamid5[valid5])
        kyle_lambda = float(cov[0, 1] / np.var(signed_vol[valid5]))
    else:
        kyle_lambda = 0.0

    vpin_mean, vpin_std = compute_vpin(side_sign, notional)

    signal_active = np.abs(logit) > LOGIT_THRESHOLD

    sig_fill_mask = (
        (~signal_active) |
        (pred == 0) |
        ((pred == 1)  & (side_sign == -1)) |
        ((pred == -1) & (side_sign == 1))
    )

    n_fills_naive   = n
    naive_spread_total = n_fills_naive * HALF_SPREAD_BPS
    naive_rebate_total = n_fills_naive * MAKER_REBATE_BPS
    naive_adv_sel_total = adv_sel.get(5, 0.0) * n_fills_naive
    naive_net_bps       = naive_spread_total + naive_rebate_total - naive_adv_sel_total

    inv_naive = np.cumsum((-side_sign) * sz)
    inv_naive_std  = float(np.std(inv_naive))
    inv_naive_max  = float(np.max(np.abs(inv_naive)))
    inv_naive_hl   = inventory_half_life(inv_naive)

    mid_shift5_full = np.roll(mid, -5); mid_shift5_full[-5:] = mid[-5:]
    dm5_full = (mid_shift5_full - mid) / mid * 10_000.0
    inv_sign_per_tick = -side_sign.astype(np.float64)
    pnl_per_tick_naive = (HALF_SPREAD_BPS + MAKER_REBATE_BPS) * np.ones(n)
    pnl_per_tick_naive -= inv_sign_per_tick * dm5_full
    sharpe_naive = sharpe(pnl_per_tick_naive, ticks_per_yr)

    fills_taken   = sig_fill_mask
    fills_avoided = ~sig_fill_mask
    n_fills_sig   = int(np.sum(fills_taken))
    n_avoided     = int(np.sum(fills_avoided))

    adv_on_avoided = inv_sign_per_tick[fills_avoided] * dm5_full[fills_avoided]
    sig_adv_fills_avoided    = int(np.sum(adv_on_avoided > 0))
    sig_adv_avoidance_rate   = sig_adv_fills_avoided / max(n_avoided, 1) * 100.0

    sig_spread_total = n_fills_sig * HALF_SPREAD_BPS
    sig_rebate_total = n_fills_sig * MAKER_REBATE_BPS
    adv_on_taken = inv_sign_per_tick[fills_taken] * dm5_full[fills_taken]
    sig_adv_sel_total = float(np.sum(adv_on_taken))
    sig_net_bps  = sig_spread_total + sig_rebate_total - sig_adv_sel_total

    sig_improvement_bps = sig_net_bps - naive_net_bps

    inv_changes_sig = np.where(fills_taken, (-side_sign) * sz, 0.0)
    inv_sig = np.cumsum(inv_changes_sig)
    inv_sig_std = float(np.std(inv_sig))
    inv_sig_max = float(np.max(np.abs(inv_sig)))
    inv_sig_hl  = inventory_half_life(inv_sig)

    pnl_per_tick_sig = np.where(fills_taken,
        HALF_SPREAD_BPS + MAKER_REBATE_BPS - inv_sign_per_tick * dm5_full,
        0.0)
    sharpe_sig  = sharpe(pnl_per_tick_sig, ticks_per_yr)
    sortino_sig = sortino(pnl_per_tick_sig, ticks_per_yr)

    cum_sig = np.cumsum(pnl_per_tick_sig)
    peak    = np.maximum.accumulate(cum_sig)
    dd_sig  = peak - cum_sig
    max_dd  = float(np.max(dd_sig))
    peak_v  = float(np.max(cum_sig))
    max_dd_pct = (max_dd / peak_v * 100.0) if peak_v > 0 else 0.0
    calmar  = (sig_net_bps / DATA_DAYS * 365.0 / 10_000.0 * 100.0) / max_dd_pct if max_dd_pct > 0 else None

    gross_positive = sig_spread_total + sig_rebate_total + max(0.0, -sig_adv_sel_total)
    avoidance_gain = max(0.0, float(np.sum(np.where(adv_on_avoided > 0, adv_on_avoided, 0.0))))
    gross_total    = sig_spread_total + sig_rebate_total + avoidance_gain
    pnl_attr_spread  = sig_spread_total / gross_total * 100.0 if gross_total > 0 else 0.0
    pnl_attr_rebate  = sig_rebate_total / gross_total * 100.0 if gross_total > 0 else 0.0
    pnl_attr_adv_cost = abs(sig_adv_sel_total) / gross_total * 100.0 if gross_total > 0 else 0.0
    pnl_attr_avoid   = avoidance_gain / gross_total * 100.0 if gross_total > 0 else 0.0

    signal_guidance_rate = float(np.mean(signal_active & (pred != 0))) * 100.0
    fill_acceptance_rate = n_fills_sig / n * 100.0
    info_ratio           = (sig_improvement_bps / abs(naive_adv_sel_total)
                            if abs(naive_adv_sel_total) > 0 else 0.0)

    elapsed = time.time() - t0
    print(
        f"[DONE]  {name} | {elapsed:.1f}s | "
        f"DA={da_nz_h1:.4f} | "
        f"VPIN={vpin_mean:.3f} | "
        f"AdvSel5={adv_sel[5]:.4f}bps | "
        f"Naive_net={naive_net_bps:+.0f}bps | "
        f"Sig_net={sig_net_bps:+.0f}bps | "
        f"Improve={sig_improvement_bps:+.0f}bps",
        flush=True,
    )

    adv_sel_out = {}
    for k in ADV_SEL_HORIZONS:
        adv_sel_out[f"adverse_sel_{k}_bps"] = round(adv_sel[k], 5)

    adv_rate_out = {}
    for k in ADV_SEL_HORIZONS:
        adv_rate_out[f"adverse_sel_rate_{k}_pct"] = round(adv_sel_rate[k], 2)

    result = {
        "model_name": name,
        "eval_ticks": n,
        "da_nz_h1": round(da_nz_h1, 4),
        "horizon_da_nz": horizon_da_nz,
        "signal_guidance_rate_pct": round(signal_guidance_rate, 2),
        "logit_threshold_used": LOGIT_THRESHOLD,
        "mean_effective_half_spread_bps": round(float(np.mean(eff_hs_bps)), 5),
        "price_impact_5_bps":    round(price_impact_k[5], 5),
        "price_impact_20_bps":   round(price_impact_k[20], 5),
        "realized_half_spread_5_bps":  round(realized_hs_k[5], 5),
        "realized_half_spread_20_bps": round(realized_hs_k[20], 5),
    }
    result.update(adv_sel_out)
    result.update(adv_rate_out)
    result.update({
        "kyle_lambda_bps_per_usd": round(kyle_lambda, 8),
        "vpin_mean": round(vpin_mean, 4),
        "vpin_std":  round(vpin_std, 4),
        "naive_fills":            n,
        "naive_spread_earn_bps":  round(naive_spread_total, 2),
        "naive_rebate_bps":       round(naive_rebate_total, 2),
        "naive_adverse_sel_bps":  round(naive_adv_sel_total, 2),
        "naive_net_pnl_bps":      round(naive_net_bps, 2),
        "naive_net_pnl_usd":      round(naive_net_bps / 10_000 * NOTIONAL_USD, 2),
        "naive_inventory_std_btc": round(inv_naive_std, 4),
        "naive_inventory_max_btc": round(inv_naive_max, 4),
        "naive_inventory_half_life_ticks": round(inv_naive_hl, 1) if inv_naive_hl > 0 else None,
        "sharpe_naive": round(sharpe_naive, 2),
        "sig_fills":              n_fills_sig,
        "sig_fills_avoided":      n_avoided,
        "sig_adverse_fills_avoided": sig_adv_fills_avoided,
        "sig_adverse_avoidance_rate_pct": round(sig_adv_avoidance_rate, 2),
        "sig_spread_earn_bps":    round(sig_spread_total, 2),
        "sig_rebate_bps":         round(sig_rebate_total, 2),
        "sig_adverse_sel_bps":    round(sig_adv_sel_total, 2),
        "sig_net_pnl_bps":        round(sig_net_bps, 2),
        "sig_net_pnl_usd":        round(sig_net_bps / 10_000 * NOTIONAL_USD, 2),
        "sig_improvement_vs_naive_bps": round(sig_improvement_bps, 2),
        "sig_inventory_std_btc":  round(inv_sig_std, 4),
        "sig_inventory_max_btc":  round(inv_sig_max, 4),
        "sig_inventory_half_life_ticks": round(inv_sig_hl, 1) if inv_sig_hl > 0 else None,
        "sharpe_signal":          round(sharpe_sig, 2),
        "sortino_signal":         round(sortino_sig, 2),
        "max_drawdown_pct":       round(max_dd_pct, 4),
        "calmar_signal":          round(calmar, 3) if calmar is not None else None,
        "pnl_attr_spread_pct":    round(pnl_attr_spread, 2),
        "pnl_attr_rebate_pct":    round(pnl_attr_rebate, 2),
        "pnl_attr_adv_cost_pct":  round(pnl_attr_adv_cost, 2),
        "pnl_attr_avoidance_pct": round(pnl_attr_avoid, 2),
        "fill_acceptance_rate_pct": round(fill_acceptance_rate, 2),
        "information_ratio":      round(info_ratio, 4),
    })
    return result


def generate_report(results: list) -> None:
    import json as _json
    with open(OUT_DIR / "test_metrics.json", "w") as f:
        _json.dump(results, f, indent=2)

    pd.DataFrame(results).sort_values("da_nz_h1", ascending=False).to_csv(
        OUT_DIR / "test_rank.csv", index=False
    )

    srt = sorted(results, key=lambda x: -x["sig_net_pnl_bps"])

    adv_h_cols = " | ".join(f"AdvSel {k}tk" for k in ADV_SEL_HORIZONS)
    adv_h_sep  = ":---:|" * len(ADV_SEL_HORIZONS)

    md = [
        "# Market Making Profitability Report - Signal-Enhanced MM",
        "",
        "**Exchange**: Hyperliquid BTC Perpetual Futures  ",
        "**Data**: December 2025 * 11,918,929 BTC trade ticks  ",
        "**Evaluation Basis**: Last 75% of ticks  ",
        "**Strategy**: Passive limit order posting guided by directional signal  ",
        "**Notional**: $100,000 USD per 1-BTC equivalent position  ",
        "",
        "## Cost & Revenue Structure",
        "",
        "| Component | Per Side | Per Round-Trip |",
        "|:---|:---:|:---:|",
        "| Maker rebate (Hyperliquid) | +1.0 bps | +2.0 bps |",
        "| Half-spread earned (quoted) | +0.5 bps | +1.0 bps |",
        "| **Structural MM earn** | **+1.5 bps** | **+3.0 bps** |",
        "| Adverse selection cost | variable | variable |",
        "",
        "---",
        "",
        "## 1. Spread & Price Impact Metrics",
        "",
        "| Model | DA (h=1) | Eff. HS (bps) | Price Impact 5tk | Price Impact 20tk | Realized HS 5tk | Realized HS 20tk |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r in srt:
        md.append(
            f"| **{r['model_name']}** "
            f"| {r['da_nz_h1']:.4f} "
            f"| {r['mean_effective_half_spread_bps']:.4f} "
            f"| {r['price_impact_5_bps']:.5f} "
            f"| {r['price_impact_20_bps']:.5f} "
            f"| {r['realized_half_spread_5_bps']:.5f} "
            f"| {r['realized_half_spread_20_bps']:.5f} |"
        )

    md += [
        "",
        "---",
        "",
        "## 2. Adverse Selection Analysis (All Horizons)",
        "",
        "Adverse selection = mean mid drift (bps) in direction against MM position after fill. Positive = adverse.",
        "",
        f"| Model | DA (h=1) | {adv_h_cols} | Rate (5tk) | VPIN | Kyle lambda |",
        f"|:---|:---:|{adv_h_sep}:---:|:---:|:---:|",
    ]
    for r in srt:
        adv_vals = " ".join(f"| {r[f'adverse_sel_{k}_bps']:.5f} " for k in ADV_SEL_HORIZONS)
        md.append(
            f"| **{r['model_name']}** "
            f"| {r['da_nz_h1']:.4f} "
            f"{adv_vals}"
            f"| {r['adverse_sel_rate_5_pct']:.1f}% "
            f"| {r['vpin_mean']:.4f} "
            f"| {r['kyle_lambda_bps_per_usd']:.2e} |"
        )

    md += [
        "",
        "---",
        "",
        "## 3. Naive MM vs Signal-Guided MM Net P&L",
        "",
        "| Model | Naive Net (bps) | Sig Net (bps) | **Improvement (bps)** | Naive Net (USD) | Sig Net (USD) | Sig Sharpe | Avoidance Rate |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r in srt:
        md.append(
            f"| **{r['model_name']}** "
            f"| {r['naive_net_pnl_bps']:+.0f} "
            f"| {r['sig_net_pnl_bps']:+.0f} "
            f"| **{r['sig_improvement_vs_naive_bps']:+.0f}** "
            f"| ${r['naive_net_pnl_usd']:+,.0f} "
            f"| ${r['sig_net_pnl_usd']:+,.0f} "
            f"| {r['sharpe_signal']:.2f} "
            f"| {r['sig_adverse_avoidance_rate_pct']:.1f}% |"
        )

    md += [
        "",
        "---",
        "",
        "## 4. Inventory Risk Metrics",
        "",
        "| Model | Naive Inv Std (BTC) | Naive Inv Max (BTC) | Naive HL (tks) | Sig Inv Std | Sig Inv Max | Sig HL (tks) | Sig Sharpe | Sig Sortino | Max DD % | Calmar |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r in srt:
        n_hl  = f"{r['naive_inventory_half_life_ticks']:.0f}" if r.get("naive_inventory_half_life_ticks") else "None"
        s_hl  = f"{r['sig_inventory_half_life_ticks']:.0f}"  if r.get("sig_inventory_half_life_ticks")  else "None"
        calmar = f"{r['calmar_signal']:.2f}" if r.get("calmar_signal") else "N/A"
        md.append(
            f"| **{r['model_name']}** "
            f"| {r['naive_inventory_std_btc']:.4f} "
            f"| {r['naive_inventory_max_btc']:.4f} "
            f"| {n_hl} "
            f"| {r['sig_inventory_std_btc']:.4f} "
            f"| {r['sig_inventory_max_btc']:.4f} "
            f"| {s_hl} "
            f"| {r['sharpe_signal']:.2f} "
            f"| {r['sortino_signal']:.2f} "
            f"| {r['max_drawdown_pct']:.2f}% "
            f"| {calmar} |"
        )

    md += [
        "",
        "---",
        "",
        "## 5. P&L Attribution (Signal-Guided MM)",
        "",
        "| Model | Spread % | Rebate % | AdvSel Cost % | Avoidance Gain % | Info Ratio | Fill Rate |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r in srt:
        md.append(
            f"| **{r['model_name']}** "
            f"| {r['pnl_attr_spread_pct']:.1f}% "
            f"| {r['pnl_attr_rebate_pct']:.1f}% "
            f"| {r['pnl_attr_adv_cost_pct']:.1f}% "
            f"| {r['pnl_attr_avoidance_pct']:.1f}% "
            f"| {r['information_ratio']:.4f} "
            f"| {r['fill_acceptance_rate_pct']:.1f}% |"
        )

    md += [
        "",
        "---",
        "",
        "## 6. Multi-Horizon Non-Zero Directional Accuracy (Last 75% Ticks)",
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
        "",
        "---",
        "",
        "## Methodology Notes",
        "",
        "1. **Evaluation basis**: Last 75% of ticks by row count after merge.",
        "2. **Fill Model**: Every trade in the BTC trade stream is a potential fill for our passive quotes.",
        "   Naive MM fills on all trades; signal-guided MM skips fills when signal opposes aggressor direction.",
        "3. **Adverse Selection (k-tick)**: mean of `inv_sign x (mid[t+k] - mid[t])` in bps.",
        "   `inv_sign` = -1 if we sold (buyer aggressed), +1 if we bought (seller aggressed).",
        "4. **VPIN**: Volume-synchronized probability of informed trading computed in rolling 200-trade buckets.",
        "5. **Kyle's Lambda**: OLS coefficient of price impact regressed on signed order flow x notional.",
        "6. **Inventory Half-Life**: AR(1) fit on inventory series. Shorter = faster mean-reversion.",
        "7. **Realized Half-Spread**: `HALF_SPREAD - price_impact` = what MM actually keeps after price moves.",
        "8. **P&L Attribution**: spread earned + rebate earned + avoidance gains - adverse selection costs.",
        "9. **Adverse Avoidance Rate**: % of skipped fills where adverse selection would have occurred.",
        "10. Signal guided by `logit` confidence; threshold = 0.0 (all non-zero predictions active).",
        "11. Notional: $100,000 USD. Inventory in BTC at mean price ~$97,000.",
        "12. Funding rate, mark-to-market margin calls, and competition from other MMs not modelled.",
    ]

    rpt = OUT_DIR / "README.md"
    with open(rpt, "w") as f:
        f.write("\n".join(md) + "\n")

    print(f"\n[SAVED] {OUT_DIR / 'test_metrics.json'}")
    print(f"[SAVED] {OUT_DIR / 'test_rank.csv'}")
    print(f"[SAVED] {rpt}")


def main():
    if not MASTER_PATH.exists():
        print(f"ERROR: {MASTER_PATH} not found. Run build_trade_master.py first.", flush=True)
        return

    print("Loading BTC trade master ...", flush=True)
    t0 = time.time()
    master = pd.read_parquet(MASTER_PATH)
    print(f"Loaded {len(master):,} BTC trades in {time.time()-t0:.1f}s", flush=True)

    print("=" * 80, flush=True)
    print("  MARKET MAKING PROFITABILITY - LAST 75% TICKS EVALUATION", flush=True)
    print("=" * 80, flush=True)

    results = []
    for name, path in MODEL_CONFIGS:
        m = compute_metrics(name, path, master)
        if m is not None:
            results.append(m)
            with open(RESULTS_DIR / f"{name.replace(' ', '_').replace('/', '_')}.json", "w") as f:
                import json as _j; _j.dump(m, f, indent=2)

    generate_report(results)

    print("\n" + "=" * 80, flush=True)
    print("  COMPLETE! Report: README.md", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
