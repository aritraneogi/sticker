import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR   = Path(__file__).resolve().parent.parent.parent.parent
SCRIPT_DIR = BASE_DIR
OUT_DIR    = Path(__file__).resolve().parent.parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

TAKER_FEE      = 2.5
HALF_SPREAD    = 0.5
LAT_SLIP       = 0.5
MAKER_REBATE   = -1.0

TAKER_SIDE     = TAKER_FEE + HALF_SPREAD + LAT_SLIP
MAKER_SIDE     = MAKER_REBATE

RT_TAKER_TAKER = 2 * TAKER_SIDE
RT_TAKER_MAKER = TAKER_SIDE + MAKER_SIDE
RT_MAKER_MAKER = 2 * MAKER_SIDE

P_WIN_SCENARIOS  = [0.1, 0.3, 0.5, 0.7, 0.9]
P_FILL_SCENARIOS = [0.01, 0.03, 0.05, 0.10]
MOVE_THRESHOLDS  = [0.5, 1.0, 2.0, 5.0, 10.0]

HORIZONS     = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500]
DATA_DAYS    = 31.0
NOTIONAL_USD = 100_000

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


def compute_latarb_metrics(name: str, pred_path: Path) -> dict | None:
    if not pred_path.exists():
        print(f"[SKIP] {name}", flush=True)
        return None

    t0 = time.time()
    print(f"[PROCESS] {name} ...", flush=True)

    df    = pd.read_parquet(pred_path).sort_values("tick_idx").reset_index(drop=True)
    n_total = len(df)
    n_75    = n_total // 4
    df      = df.iloc[n_75:].reset_index(drop=True)

    mids  = df["mid"].to_numpy(dtype=np.float64)
    preds = df["pred"].to_numpy(dtype=np.int8)
    n     = len(preds)

    ret     = (mids[1:] - mids[:-1]) / mids[:-1] * 10_000.0
    abs_ret = np.abs(ret)
    pos     = preds[:-1]
    n_ticks = n - 1

    ticks_per_yr = n_ticks / DATA_DAYS * 365.0

    actual = np.sign(ret)
    nz     = actual != 0
    da     = float(np.mean(pos[nz] == actual[nz])) if nz.sum() > 0 else 0.5
    edge   = 2 * da - 1

    gross_per_tick   = pos * ret
    mean_gross_tick  = float(np.mean(gross_per_tick))
    total_gross_bps  = float(np.sum(gross_per_tick))

    mean_abs_ret = float(np.mean(abs_ret))
    q50_abs      = float(np.percentile(abs_ret, 50))
    q90_abs      = float(np.percentile(abs_ret, 90))
    q95_abs      = float(np.percentile(abs_ret, 95))
    q99_abs      = float(np.percentile(abs_ret, 99))

    prev_pos         = np.concatenate([[0], pos[:-1]])
    changed          = pos != prev_pos
    n_signal_changes = int(np.sum(changed))
    avg_hold_ticks   = n_ticks / n_signal_changes if n_signal_changes > 0 else n_ticks
    gross_per_trade  = total_gross_bps / n_signal_changes if n_signal_changes > 0 else 0.0

    tt_results = {}
    for p_win in P_WIN_SCENARIOS:
        net_per_attempt = p_win * (gross_per_trade - RT_TAKER_TAKER)
        n_fills         = round(n_signal_changes * p_win)
        total_net_bps   = n_signal_changes * net_per_attempt
        total_net_usd   = total_net_bps / 10_000.0 * NOTIONAL_USD
        ann_net_pct     = (net_per_attempt / n_ticks * ticks_per_yr) / 10_000.0 * 100.0
        tt_results[f"p_win_{p_win}"] = {
            "n_fills":          n_fills,
            "net_per_attempt_bps": round(net_per_attempt, 5),
            "total_net_bps":    round(total_net_bps, 0),
            "total_net_usd":    round(total_net_usd, 2),
            "annualized_net_pct": round(ann_net_pct, 3),
            "profitable":       net_per_attempt > 0,
        }

    tm_results = {}
    for p_win in P_WIN_SCENARIOS:
        net_per_attempt = p_win * (gross_per_trade - RT_TAKER_MAKER)
        n_fills         = round(n_signal_changes * p_win)
        total_net_bps   = n_signal_changes * net_per_attempt
        total_net_usd   = total_net_bps / 10_000.0 * NOTIONAL_USD
        ann_net_pct     = (net_per_attempt / n_ticks * ticks_per_yr) / 10_000.0 * 100.0
        tm_results[f"p_win_{p_win}"] = {
            "n_fills":          n_fills,
            "net_per_attempt_bps": round(net_per_attempt, 5),
            "total_net_bps":    round(total_net_bps, 0),
            "total_net_usd":    round(total_net_usd, 2),
            "annualized_net_pct": round(ann_net_pct, 3),
            "profitable":       net_per_attempt > 0,
        }

    structural_per_fill     = (-RT_MAKER_MAKER) + (2 * HALF_SPREAD)
    signal_alpha_per_fill   = edge * mean_abs_ret
    net_per_fill_mm         = structural_per_fill + signal_alpha_per_fill
    net_per_fill_mm_no_sig  = structural_per_fill

    mm_results = {}
    for p_fill in P_FILL_SCENARIOS:
        n_fills       = round(n_ticks * p_fill)
        total_net_bps = n_fills * net_per_fill_mm
        total_net_usd = total_net_bps / 10_000.0 * NOTIONAL_USD
        ann_rate      = p_fill * ticks_per_yr * net_per_fill_mm
        ann_net_pct   = ann_rate / 10_000.0 * 100.0
        pnl_ticks     = np.where(
            np.random.random(n_ticks) < p_fill,
            net_per_fill_mm, 0.0
        )
        mu_t = np.mean(pnl_ticks); sd_t = np.std(pnl_ticks)
        sharpe_mm = float((mu_t / sd_t) * np.sqrt(ticks_per_yr)) if sd_t > 0 else 0.0
        mm_results[f"p_fill_{p_fill}"] = {
            "n_fills":          n_fills,
            "net_per_fill_bps": round(net_per_fill_mm, 4),
            "net_per_fill_bps_no_signal": round(net_per_fill_mm_no_sig, 4),
            "signal_alpha_per_fill_bps": round(signal_alpha_per_fill, 5),
            "total_net_bps":    round(total_net_bps, 0),
            "total_net_usd":    round(total_net_usd, 2),
            "annualized_net_pct": round(ann_net_pct, 2),
            "approx_sharpe":    round(sharpe_mm, 1),
            "profitable":       net_per_fill_mm > 0,
        }

    sel_results = {}
    for threshold in MOVE_THRESHOLDS:
        mask   = abs_ret > threshold
        n_opps = int(np.sum(mask))
        if n_opps == 0:
            sel_results[f"T{threshold}bps"] = {"n_opportunities": 0}
            continue
        frac_opps    = n_opps / n_ticks
        mean_abs_t   = float(np.mean(abs_ret[mask]))
        gross_per_opp = edge * mean_abs_t
        net_per_opp_tt = gross_per_opp - RT_TAKER_TAKER
        net_per_opp_tm = gross_per_opp - RT_TAKER_MAKER
        p_win_be_tt  = (RT_TAKER_TAKER / gross_per_opp) if gross_per_opp > 0 else None
        p_win_be_tm  = (RT_TAKER_MAKER / gross_per_opp) if gross_per_opp > 0 else None
        total_gross_filtered = float(np.sum((pos * ret)[mask]))
        total_net_tt = total_gross_filtered - n_opps * RT_TAKER_TAKER
        total_net_tm = total_gross_filtered - n_opps * RT_TAKER_MAKER
        sel_results[f"T{threshold}bps"] = {
            "n_opportunities":          n_opps,
            "frac_of_ticks_pct":        round(frac_opps * 100.0, 3),
            "mean_abs_ret_bps":         round(mean_abs_t, 4),
            "gross_per_opp_bps":        round(gross_per_opp, 5),
            "net_per_opp_taker_taker":  round(net_per_opp_tt, 5),
            "net_per_opp_taker_maker":  round(net_per_opp_tm, 5),
            "p_win_breakeven_tt":       round(p_win_be_tt, 3) if p_win_be_tt is not None else None,
            "p_win_breakeven_tm":       round(p_win_be_tm, 3) if p_win_be_tm is not None else None,
            "total_net_taker_taker_bps": round(total_net_tt, 0),
            "total_net_taker_maker_bps": round(total_net_tm, 0),
            "tt_profitable_at_p1":      net_per_opp_tt > 0,
            "tm_profitable_at_p1":      net_per_opp_tm > 0,
        }

    be_p_win_tt = RT_TAKER_TAKER / gross_per_trade if gross_per_trade > 0 else None
    be_p_win_tm = RT_TAKER_MAKER / gross_per_trade if gross_per_trade > 0 else None
    be_rt_bps   = gross_per_trade

    ms_mid = pd.Series(mids)
    horizon_da_nz = {}
    for h in HORIZONS:
        diff_h = ms_mid.shift(-h).values - mids
        act_h  = np.where(diff_h > 1e-8, 1, np.where(diff_h < -1e-8, -1, 0))
        nz_h   = ~np.isnan(diff_h) & (act_h != 0)
        horizon_da_nz[f"h{h}"] = float(np.mean(act_h[nz_h] == preds[nz_h])) if nz_h.sum() > 0 else None

    elapsed = time.time() - t0
    print(
        f"[DONE]  {name} | {elapsed:.1f}s | "
        f"DA={da:.4f} | Edge={edge:.4f} | "
        f"GrossPerTrade={gross_per_trade:.3f}bps | "
        f"MM_net_per_fill={net_per_fill_mm:.3f}bps",
        flush=True,
    )

    return {
        "model_name":           name,
        "eval_ticks":           n,
        "da_nz_h1":             round(da, 4),
        "horizon_da_nz":        horizon_da_nz,
        "signal_edge":          round(edge, 4),
        "total_gross_bps":      round(total_gross_bps, 2),
        "mean_gross_per_tick":  round(mean_gross_tick, 6),
        "n_signal_changes":     n_signal_changes,
        "avg_hold_ticks":       round(avg_hold_ticks, 2),
        "gross_per_trade_bps":  round(gross_per_trade, 4),
        "breakeven_rt_bps":     round(be_rt_bps, 4),
        "mean_abs_ret_bps":     round(mean_abs_ret, 5),
        "q50_abs_ret_bps":      round(q50_abs, 5),
        "q90_abs_ret_bps":      round(q90_abs, 5),
        "q95_abs_ret_bps":      round(q95_abs, 5),
        "q99_abs_ret_bps":      round(q99_abs, 5),
        "breakeven_p_win_taker_taker": round(be_p_win_tt, 3) if be_p_win_tt is not None else None,
        "breakeven_p_win_taker_maker": round(be_p_win_tm, 3) if be_p_win_tm is not None else None,
        "mm_structural_per_fill_bps": round(structural_per_fill, 4),
        "mm_signal_alpha_per_fill_bps": round(signal_alpha_per_fill, 5),
        "mm_net_per_fill_bps":    round(net_per_fill_mm, 4),
        "mm_net_per_fill_no_signal_bps": round(net_per_fill_mm_no_sig, 4),
        "mm_signal_improvement_pct": round(
            (net_per_fill_mm - net_per_fill_mm_no_sig) / net_per_fill_mm_no_sig * 100.0, 2
        ) if net_per_fill_mm_no_sig != 0 else None,
        "taker_taker_arb":      tt_results,
        "taker_maker_arb":      tm_results,
        "market_making":        mm_results,
        "selective_arb":        sel_results,
    }


def generate_report(results: list) -> None:
    with open(OUT_DIR / "test_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    rows = []
    for r in results:
        base = {k: v for k, v in r.items()
                if k not in ("taker_taker_arb", "taker_maker_arb", "market_making", "selective_arb", "horizon_da_nz")}
        mm = r.get("market_making", {}).get("p_fill_0.05", {})
        for k, v in mm.items():
            base[f"mm_pfill05__{k}"] = v
        rows.append(base)
    pd.DataFrame(rows).sort_values("da_nz_h1", ascending=False).to_csv(
        OUT_DIR / "test_rank.csv", index=False
    )

    md = [
        "# Latency Arbitrage & Market Making Profitability Report",
        "",
        "**Dataset**: 11,918,929 Hyperliquid BTC Perpetual Futures ticks (Dec 2025)  ",
        "**Evaluation Basis**: Last 75% of ticks  ",
        "**Notional**: $100,000 USD per position  ",
        "",
        "## Cost Structure",
        "",
        "| Execution Mode | Entry Cost | Exit Cost | Round-Trip Total |",
        "|:---|:---:|:---:|:---:|",
        "| Taker-Taker (scalp / LA aggressive) | +3.5 bps | +3.5 bps | **+7.0 bps** |",
        "| Taker-Maker (LA with passive exit) | +3.5 bps | -1.0 bps | **+2.5 bps** |",
        "| Maker-Maker (Market Making) | -1.0 bps | -1.0 bps | **-2.0 bps** (earn) |",
        "",
        "---",
        "",
        "## 1. Breakeven Conditions",
        "",
        "| Model | DA (h=1) | Signal Edge | Gross/Trade (bps) | BE RT Cost (bps) | BE p_win (TT 7bps) | BE p_win (TM 2.5bps) |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r in sorted(results, key=lambda x: -x["da_nz_h1"]):
        be_tt = f"{r['breakeven_p_win_taker_taker']:.2f}" if r.get("breakeven_p_win_taker_taker") is not None else "N/A"
        be_tm = f"{r['breakeven_p_win_taker_maker']:.2f}" if r.get("breakeven_p_win_taker_maker") is not None else "N/A"
        md.append(
            f"| **{r['model_name']}** "
            f"| {r['da_nz_h1']:.4f} "
            f"| {r['signal_edge']:.4f} "
            f"| {r['gross_per_trade_bps']:.3f} "
            f"| **{r['breakeven_rt_bps']:.3f}** "
            f"| {be_tt} "
            f"| {be_tm} |"
        )

    md += [
        "",
        "---",
        "",
        "## 2. Market Making Net P&L (signal-enhanced, maker-maker execution)",
        "",
        "| Model | DA | MM Net/Fill (bps) | No-Signal Baseline | Signal Uplift | Ann. Net % (p_fill=5%) | Ann. Net USD (p_fill=5%) |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r in sorted(results, key=lambda x: -x["mm_net_per_fill_bps"]):
        mm5   = r.get("market_making", {}).get("p_fill_0.05", {})
        ann5  = f"{mm5.get('annualized_net_pct', 0):+.1f}%"
        usd5  = f"${mm5.get('total_net_usd', 0):+,.0f}"
        uplift = f"{r.get('mm_signal_improvement_pct', 0):+.2f}%" if r.get("mm_signal_improvement_pct") is not None else "N/A"
        md.append(
            f"| **{r['model_name']}** "
            f"| {r['da_nz_h1']:.4f} "
            f"| **{r['mm_net_per_fill_bps']:.4f}** "
            f"| {r['mm_net_per_fill_no_signal_bps']:.4f} "
            f"| {uplift} "
            f"| {ann5} "
            f"| {usd5} |"
        )

    md += [
        "",
        "---",
        "",
        "## 3. Multi-Horizon Non-Zero DA (Last 75% Ticks)",
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
        "- Evaluation basis: Last 75% of ticks by row count.",
        "- Taker-Taker/Maker P&L = p_win x (gross_per_signal_change - RT_cost) x n_signal_changes",
        "- MM structural = 2 x half_spread_earned (0.5 bps/side) + 2 x maker_rebate (1.0 bps/side) = 3.0 bps/RT",
        "- MM signal_alpha = (2xDA - 1) x mean_abs_tick_return",
        "- Selective arb assumes perfect ex-ante move-size filtering (upper bound)",
        "- Funding rate, margin, inventory risk, and operational costs not included",
    ]

    rpt = OUT_DIR / "README.md"
    with open(rpt, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"\n[SAVED] {OUT_DIR / 'test_metrics.json'}")
    print(f"[SAVED] {OUT_DIR / 'test_rank.csv'}")
    print(f"[SAVED] {rpt}")


def main() -> None:
    np.random.seed(42)
    print("=" * 85, flush=True)
    print("  LATENCY ARBITRAGE & MARKET MAKING - LAST 75% TICKS EVALUATION")
    print(f"  RT costs: TT={RT_TAKER_TAKER}bps | TM={RT_TAKER_MAKER}bps | MM={RT_MAKER_MAKER}bps (earn)")
    print("=" * 85, flush=True)

    results = []
    for name, path in MODEL_CONFIGS:
        m = compute_latarb_metrics(name, path)
        if m is not None:
            results.append(m)

    generate_report(results)

    print("\n" + "=" * 85, flush=True)
    print("  COMPLETE!")
    print(f"  Report: {OUT_DIR / 'README.md'}")
    print("=" * 85, flush=True)


if __name__ == "__main__":
    main()
