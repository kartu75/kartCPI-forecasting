"""
ARDL Lag Selection Script
-------------------------
Compares three pre-specified ARDL candidate models for forecasting Headline CPI.
Dependent variable : Delta(Headline_CPI)
Exogenous variables: Delta(Food_and_Beverages), Delta(Fuel_and_Light)

Candidate models tested:
  Candidate A: ARDL(1, 1, 1) — 1 AR lag for CPI, 1 lag each for Food & Fuel
  Candidate B: ARDL(2, 2, 2) — 2 AR lags for CPI, 2 lags each for Food & Fuel
  Candidate C: ARDL(3, 3, 3) — 3 AR lags for CPI, 3 lags each for Food & Fuel

The best model is selected based on lowest AIC (primary) and BIC (secondary).
Prints a full comparison table and announces the winner.
"""

import os
import sys
import pandas as pd
import numpy as np
from statsmodels.tsa.ardl import ARDL
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ── Locate processed data ─────────────────────────────────────────────────────
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "data", "processed_data.csv")

if not os.path.exists(data_path):
    print(f"ERROR: processed_data.csv not found at {data_path}")
    print("Please run data_prep.py first.")
    sys.exit(1)

df = pd.read_csv(data_path, index_col=0, parse_dates=True)

# ── First differences (variables are I(1)) ────────────────────────────────────
full_diff = df.diff().dropna()

# Train / Test split identical to the main pipeline
train_diff = full_diff.loc[:'2023-12-01']
test_diff  = full_diff.loc['2024-01-01':]

last_train_cpi = df.loc[:'2023-12-01', 'Headline_CPI'].iloc[-1]

endog_train = train_diff['Headline_CPI']
exog_train  = train_diff[['Food_and_Beverages', 'Fuel_and_Light']]
exog_test   = test_diff[['Food_and_Beverages', 'Fuel_and_Light']]
test_levels = df.loc['2024-01-01':, 'Headline_CPI']

# ── Candidate specifications ──────────────────────────────────────────────────
# Each entry: (label, ar_lags_for_CPI, lags_for_Food, lags_for_Fuel)
candidates = [
    ("Candidate A: ARDL(1,1,1)", 1, {"Food_and_Beverages": 1, "Fuel_and_Light": 1}),
    ("Candidate B: ARDL(2,2,2)", 2, {"Food_and_Beverages": 2, "Fuel_and_Light": 2}),
    ("Candidate C: ARDL(3,3,3)", 3, {"Food_and_Beverages": 3, "Fuel_and_Light": 3}),
]

# ── Evaluate each candidate ───────────────────────────────────────────────────
results = []

for label, ar_lags, exog_orders in candidates:
    model = ARDL(endog_train, lags=ar_lags, exog=exog_train, order=exog_orders, trend='c')
    fitted = model.fit()

    aic = fitted.aic
    bic = fitted.bic

    # Out-of-sample forecast using actual test exog values
    fc_diff = fitted.forecast(steps=len(test_diff), exog=exog_test)
    fc_levels = last_train_cpi + np.cumsum(fc_diff.values)

    rmse = np.sqrt(mean_squared_error(test_levels, fc_levels))
    mae  = mean_absolute_error(test_levels, fc_levels)

    results.append({
        "Label"  : label,
        "AR Lags": ar_lags,
        "AIC"    : aic,
        "BIC"    : bic,
        "RMSE"   : rmse,
        "MAE"    : mae,
    })
    print(f"  Evaluated {label}  |  AIC={aic:.2f}  BIC={bic:.2f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

# ── Print comparison table ────────────────────────────────────────────────────
res_df = pd.DataFrame(results).set_index("Label")

print("\n" + "="*80)
print("  ARDL CANDIDATE MODEL COMPARISON")
print("="*80)
print(res_df[["AR Lags","AIC","BIC","RMSE","MAE"]].to_string())
print("="*80)

# Select winner: lowest AIC (primary), then lowest BIC on tie
best_idx = res_df["AIC"].idxmin()
winner   = res_df.loc[best_idx]
print(f"\n  WINNER (lowest AIC): {best_idx}")
print(f"    AR Lags = {int(winner['AR Lags'])} | AIC = {winner['AIC']:.2f} | BIC = {winner['BIC']:.2f}")
print(f"    RMSE = {winner['RMSE']:.4f} | MAE = {winner['MAE']:.4f}")
print("="*80)
