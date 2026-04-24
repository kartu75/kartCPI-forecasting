import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def evaluate_models(df, results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    report_lines = []
    
    # Train-Test Split (Train: 2015-2023, Test: 2024-2025)
    train = df.loc[:'2023-12-01']
    test = df.loc['2024-01-01':]
    
    # We will model on First Differences because they are mostly I(1)
    train_diff = train.diff().dropna()
    test_diff = test.diff().dropna()
    # Align test size (since diff drops first row, if test has 24 rows, test_diff has 23 unless we diff full and then split)
    full_diff = df.diff().dropna()
    train_diff = full_diff.loc[:'2023-12-01']
    test_diff = full_diff.loc['2024-01-01':]
    
    report_lines.append("=== MODEL EVALUATION (Train: 2015-2023, Test: 2024-2025) ===\n")
    
    # --- MODEL 1: Benchmark OLS (single-equation) ---
    # Dependent: ΔCPI_t
    # Independents: CPI_{t-1} (lagged), Food_t and Fuel_t (contemporaneous)
    last_train_cpi = train['Headline_CPI'].iloc[-1]

    def build_m1_XY(data):
        """Build design matrix for Model 1: CPI at lag-1; Food & Fuel at time t."""
        d_cpi  = data['Headline_CPI'].diff()
        cpi_l1 = data['Headline_CPI'].shift(1)
        food_t = data['Food_and_Beverages']
        fuel_t = data['Fuel_and_Light']
        frame = pd.concat([d_cpi, cpi_l1, food_t, fuel_t], axis=1)
        frame.columns = ['delta_cpi', 'cpi_lag1', 'food_t', 'fuel_t']
        return frame.dropna()

    m1_train = build_m1_XY(train)
    Y1 = m1_train['delta_cpi']
    X1 = sm.add_constant(m1_train[['cpi_lag1', 'food_t', 'fuel_t']])
    res1 = sm.OLS(Y1, X1).fit()

    # Build test regressors: CPI lag uses last train value; Food & Fuel are actual test values
    m1_test = build_m1_XY(pd.concat([train.iloc[[-1]], test]))
    X1_test = sm.add_constant(m1_test[['cpi_lag1', 'food_t', 'fuel_t']])
    fc1_delta = res1.predict(X1_test)
    fc1_cpi = last_train_cpi + np.cumsum(fc1_delta.values)

    rmse1 = np.sqrt(mean_squared_error(test['Headline_CPI'].iloc[:len(fc1_cpi)], fc1_cpi))
    mae1  = mean_absolute_error(test['Headline_CPI'].iloc[:len(fc1_cpi)], fc1_cpi)
    report_lines.append("Model 1: Benchmark OLS — ΔCPI_t ~ const + CPI_{t-1} + Food_t + Fuel_t")
    report_lines.append(f"  RMSE: {rmse1:.4f}")
    report_lines.append(f"  MAE: {mae1:.4f}\n")

    # --- MODEL 2: Proposed OLS (single-equation, CPI in levels) ---
    # Dependent: CPI_t
    # Independents: CPI_{t-1} (lagged); Food_t, Fuel_t, and all macro predictors at time t (contemporaneous)
    m2_other_cols = ['Food_and_Beverages', 'Fuel_and_Light', 'Miscellaneous', 'Housing', 'Log_USD_INR',
                     'Repo_Rate', 'Log_Reserve_Money', 'Log_Trade_Balance', 'Log_Forex_Reserves']

    def build_m2_XY(data):
        """Build design matrix for Model 2: CPI lag-1; Food, Fuel & all other predictors at time t."""
        y      = data['Headline_CPI']
        cpi_l1 = data['Headline_CPI'].shift(1).rename('Headline_CPI_lag1')
        others = data[m2_other_cols]          # contemporaneous
        frame  = pd.concat([y, cpi_l1, others], axis=1).dropna()
        return frame

    m2_train = build_m2_XY(train)
    Y2 = m2_train['Headline_CPI']
    X2_cols = ['Headline_CPI_lag1'] + m2_other_cols
    X2 = sm.add_constant(m2_train[X2_cols])
    res2 = sm.OLS(Y2, X2).fit()

    m2_test_all = build_m2_XY(pd.concat([train.iloc[[-1]], test]))
    X2_test = sm.add_constant(m2_test_all[X2_cols])
    fc2_cpi = res2.predict(X2_test).values

    rmse2 = np.sqrt(mean_squared_error(test['Headline_CPI'].iloc[:len(fc2_cpi)], fc2_cpi))
    mae2  = mean_absolute_error(test['Headline_CPI'].iloc[:len(fc2_cpi)], fc2_cpi)
    report_lines.append("Model 2: Proposed OLS — CPI_t ~ const + CPI_{t-1} + Food_t + Fuel_t + Misc_t + Housing_t + LogUSDINR_t + Repo_t + LogResM_t + LogTradeBal_t + LogForex_t")
    report_lines.append(f"  RMSE: {rmse2:.4f}")
    report_lines.append(f"  MAE: {mae2:.4f}\n")
    
    # --- MODEL 3: ARDL (Autoregressive Distributed Lag) ---
    # Three candidate specifications are evaluated; the best is chosen by AIC.
    #   Candidate A: ARDL(1,1,1) — 1 AR lag for CPI, 1 distributed lag each for Food & Fuel
    #   Candidate B: ARDL(2,2,2) — 2 AR lags for CPI, 2 distributed lags each for Food & Fuel
    #   Candidate A: ARDL(1,1,1) — AR lag=1, 1 distributed lag each for Food & Fuel
    #   Candidate B: ARDL(1,2,2) — AR lag=1, 2 distributed lags each for Food & Fuel
    #   Candidate C: ARDL(1,3,3) — AR lag=1, 3 distributed lags each for Food & Fuel
    from statsmodels.tsa.ardl import ARDL

    endog_train = train_diff['Headline_CPI']
    exog_train  = train_diff[['Food_and_Beverages', 'Fuel_and_Light']]
    exog_test   = test_diff[['Food_and_Beverages', 'Fuel_and_Light']]

    ardl_candidates = [
        ("ARDL(1,1,1)", 1, {"Food_and_Beverages": 1, "Fuel_and_Light": 1}),
        ("ARDL(1,2,2)", 1, {"Food_and_Beverages": 2, "Fuel_and_Light": 2}),
        ("ARDL(1,3,3)", 1, {"Food_and_Beverages": 3, "Fuel_and_Light": 3}),
    ]

    report_lines.append("--- Model 3: ARDL Candidate Comparison (dependent: Delta CPI | exog: Food, Fuel) ---")
    report_lines.append(f"  {'Specification':<18} {'AIC':>10} {'BIC':>10} {'RMSE':>10} {'MAE':>10}")
    report_lines.append("  " + "-"*58)

    best_aic    = np.inf
    best_label  = None
    best_fc     = None
    best_res3   = None

    for label, ar_lags, exog_orders in ardl_candidates:
        m = ARDL(endog_train, lags=ar_lags, exog=exog_train, order=exog_orders, trend='c')
        r = m.fit()
        fc_diff = r.forecast(steps=len(test_diff), exog=exog_test)
        fc_lvl  = last_train_cpi + np.cumsum(fc_diff.values)
        rmse_c  = np.sqrt(mean_squared_error(test['Headline_CPI'], fc_lvl))
        mae_c   = mean_absolute_error(test['Headline_CPI'], fc_lvl)
        report_lines.append(
            f"  {label:<18} {r.aic:>10.4f} {r.bic:>10.4f} {rmse_c:>10.4f} {mae_c:>10.4f}"
        )
        if r.aic < best_aic:
            best_aic   = r.aic
            best_label = label
            best_fc    = fc_lvl
            best_res3  = r

    report_lines.append("  " + "-"*58)
    report_lines.append(f"  Selected: {best_label}  (lowest AIC = {best_aic:.4f})\n")

    fc3_cpi = best_fc
    rmse3   = np.sqrt(mean_squared_error(test['Headline_CPI'], fc3_cpi))
    mae3    = mean_absolute_error(test['Headline_CPI'], fc3_cpi)
    report_lines.append(f"Model 3: {best_label} — ARDL (Headline CPI with Food and Fuel distributed lags)")
    report_lines.append(f"  RMSE: {rmse3:.4f}")
    report_lines.append(f"  MAE: {mae3:.4f}\n")

    # --- MODEL 4: ARDL with Controls (Food, Fuel + Macro Variables) ---
    # Extends Model 3 by adding macro controls as additional exog regressors.
    # Controls: Log_USD_INR, Repo_Rate, Log_Reserve_Money, Log_Trade_Balance, Log_Forex_Reserves
    # Two candidates tested — CPI lag=1, food/fuel lag=1, controls lag varies (0, 1).
    #   Candidate A: ARDL(1 | food=1, fuel=1, controls=0)
    #   Candidate B: ARDL(1 | food=1, fuel=1, controls=1)

    ctrl_cols = ['Log_USD_INR', 'Repo_Rate', 'Log_Reserve_Money', 'Log_Trade_Balance', 'Log_Forex_Reserves']
    exog4_train = train_diff[['Food_and_Beverages', 'Fuel_and_Light'] + ctrl_cols]
    exog4_test  = test_diff[['Food_and_Beverages', 'Fuel_and_Light'] + ctrl_cols]

    def _ctrl_order(food_fuel_lag, ctrl_lag):
        """Build an exog order dict: food/fuel use food_fuel_lag, controls use ctrl_lag."""
        od = {c: food_fuel_lag for c in ['Food_and_Beverages', 'Fuel_and_Light']}
        od.update({c: ctrl_lag for c in ctrl_cols})
        return od

    ardl4_candidates = [
        ("ARDL4(1,1,0ctrl)", 1, _ctrl_order(1, 0)),
        ("ARDL4(1,1,1ctrl)", 1, _ctrl_order(1, 1)),
    ]

    report_lines.append("--- Model 4: ARDL+Controls Candidate Comparison (exog: Food, Fuel, USD_INR, Repo, ReserveM, TradeBal, Forex) ---")
    report_lines.append(f"  {'Specification':<22} {'AIC':>10} {'BIC':>10} {'RMSE':>10} {'MAE':>10}")
    report_lines.append("  " + "-"*62)

    best4_aic   = np.inf
    best4_label = None
    best4_fc    = None

    for label, ar_lags, exog_orders in ardl4_candidates:
        m4 = ARDL(endog_train, lags=ar_lags, exog=exog4_train, order=exog_orders, trend='c')
        r4 = m4.fit()
        fc4_diff = r4.forecast(steps=len(test_diff), exog=exog4_test)
        fc4_lvl  = last_train_cpi + np.cumsum(fc4_diff.values)
        rmse_c4  = np.sqrt(mean_squared_error(test['Headline_CPI'], fc4_lvl))
        mae_c4   = mean_absolute_error(test['Headline_CPI'], fc4_lvl)
        report_lines.append(
            f"  {label:<22} {r4.aic:>10.4f} {r4.bic:>10.4f} {rmse_c4:>10.4f} {mae_c4:>10.4f}"
        )
        if r4.aic < best4_aic:
            best4_aic   = r4.aic
            best4_label = label
            best4_fc    = fc4_lvl
            best4_order = exog_orders
            best4_arlags = ar_lags

    report_lines.append("  " + "-"*62)
    report_lines.append(f"  Selected: {best4_label}  (lowest AIC = {best4_aic:.4f})\n")

    fc4_cpi = best4_fc
    rmse4   = np.sqrt(mean_squared_error(test['Headline_CPI'], fc4_cpi))
    mae4    = mean_absolute_error(test['Headline_CPI'], fc4_cpi)
    report_lines.append(f"Model 4: {best4_label} — ARDL with Controls (Food, Fuel + Macro Variables)")
    report_lines.append(f"  RMSE: {rmse4:.4f}")
    report_lines.append(f"  MAE: {mae4:.4f}\n")

    with open(os.path.join(results_dir, "model_evaluation_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("Saved model evaluation reports to results/model_evaluation_report.txt")

    # Return metrics for reference
    return {
        'Model1': {'RMSE': rmse1, 'MAE': mae1},
        'Model2': {'RMSE': rmse2, 'MAE': mae2},
        'Model3': {'RMSE': rmse3, 'MAE': mae3},
        'Model4': {'RMSE': rmse4, 'MAE': mae4},
        '_ardl4_best_ar': best4_arlags,
        '_ardl4_best_order': best4_order,
    }

if __name__ == "__main__":
    df = pd.read_csv("c:/Users/tarun/BITS/3-2/AE/project3/CPI-forecasting/data/processed_data.csv", index_col=0, parse_dates=True)
    evaluate_models(df, "c:/Users/tarun/BITS/3-2/AE/project3/CPI-forecasting/results")
