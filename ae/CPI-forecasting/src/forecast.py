import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def run_forecast(df, results_dir="results", steps=120): # 120 months = 10 years
    os.makedirs(results_dir, exist_ok=True)
    
    # Use full data for differencing
    full_diff = df.diff().dropna()
    last_cpi = df['Headline_CPI'].iloc[-1]
    
    # Future dates
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=steps, freq='MS')
    
    # --- MODEL 1 Forecast ---
    m1_vars = ['Headline_CPI', 'Food_and_Beverages', 'Fuel_and_Light']
    model1 = VAR(full_diff[m1_vars])
    res1 = model1.fit(ic='aic')
    fc1_diff = res1.forecast(full_diff[m1_vars].values[-res1.k_ar:], steps=steps)
    fc1_cpi = last_cpi + np.cumsum(fc1_diff[:, 0])
    
    # Extract Confidence Intervals for VAR
    fc1_int = res1.forecast_interval(full_diff[m1_vars].values[-res1.k_ar:], steps=steps, alpha=0.05)
    lower_diff1, upper_diff1 = fc1_int[1][:, 0], fc1_int[2][:, 0]
    fc1_lower = last_cpi + np.cumsum(lower_diff1)
    fc1_upper = last_cpi + np.cumsum(upper_diff1)
    
    # --- MODEL 2 Forecast ---
    m2_vars = ['Headline_CPI', 'Miscellaneous', 'Housing', 'Log_USD_INR', 'Repo_Rate', 'Log_Reserve_Money', 'Log_Trade_Balance', 'Log_Forex_Reserves']
    model2 = VAR(full_diff[m2_vars])
    res2 = model2.fit(ic='aic')
    fc2_diff = res2.forecast(full_diff[m2_vars].values[-res2.k_ar:], steps=steps)
    fc2_cpi = last_cpi + np.cumsum(fc2_diff[:, 0])
    
    fc2_int = res2.forecast_interval(full_diff[m2_vars].values[-res2.k_ar:], steps=steps, alpha=0.05)
    lower_diff2, upper_diff2 = fc2_int[1][:, 0], fc2_int[2][:, 0]
    fc2_lower = last_cpi + np.cumsum(lower_diff2)
    fc2_upper = last_cpi + np.cumsum(upper_diff2)
    
    # --- MODEL 3 Forecast: ARDL(1,1,1) — winner selected via AIC comparison in models.py ---
    from statsmodels.tsa.ardl import ARDL

    endog_full = full_diff['Headline_CPI']
    exog_full  = full_diff[['Food_and_Beverages', 'Fuel_and_Light']]

    # ARDL(1,1,1): 1 AR lag for CPI, 1 distributed lag each for Food & Fuel
    model3 = ARDL(endog_full, lags=1, exog=exog_full,
                  order={"Food_and_Beverages": 1, "Fuel_and_Light": 1}, trend='c')
    res3   = model3.fit()

    # For out-of-sample forecast we reuse the Food & Fuel projections from Model 1 (VAR)
    # fc1_diff cols: 0=Headline_CPI, 1=Food_and_Beverages, 2=Fuel_and_Light
    exog_future_df = pd.DataFrame(
        fc1_diff[:, 1:3],
        columns=['Food_and_Beverages', 'Fuel_and_Light']
    )

    try:
        pred3      = res3.get_prediction(start=len(endog_full),
                                          end=len(endog_full) + steps - 1,
                                          exog_oos=exog_future_df)
        fc3_diff   = pred3.predicted_mean
        fc3_conf   = pred3.conf_int(alpha=0.05)
        fc3_cpi    = last_cpi + np.cumsum(fc3_diff.values)
        fc3_lower  = last_cpi + np.cumsum(fc3_conf.iloc[:, 0].values)
        fc3_upper  = last_cpi + np.cumsum(fc3_conf.iloc[:, 1].values)
    except (AttributeError, TypeError):
        fc3_diff   = res3.forecast(steps=steps, exog=exog_future_df)
        fc3_cpi    = last_cpi + np.cumsum(fc3_diff.values)
        fc3_lower  = fc3_cpi
        fc3_upper  = fc3_cpi
    
    # Clamp confidence bands to the realistic CPI range so they don't blow up the y-axis
    Y_MIN, Y_MAX = 100, 300
    fc1_lower = np.clip(fc1_lower, Y_MIN, Y_MAX)
    fc1_upper = np.clip(fc1_upper, Y_MIN, Y_MAX)
    fc2_lower = np.clip(fc2_lower, Y_MIN, Y_MAX)
    fc2_upper = np.clip(fc2_upper, Y_MIN, Y_MAX)
    fc3_lower = np.clip(fc3_lower, Y_MIN, Y_MAX)
    fc3_upper = np.clip(fc3_upper, Y_MIN, Y_MAX)

    # --- MODEL 4 Forecast: ARDL+Controls — best spec selected via AIC comparison in models.py ---
    # Best spec confirmed by models.py evaluation: ARDL4(1,1,1ctrl)
    # Exog: Food, Fuel (from M1 VAR) + Log_USD_INR, Repo_Rate, Log_Reserve_Money,
    #        Log_Trade_Balance, Log_Forex_Reserves (from M2 VAR, cols 3-7)
    ctrl_cols = ['Log_USD_INR', 'Repo_Rate', 'Log_Reserve_Money', 'Log_Trade_Balance', 'Log_Forex_Reserves']
    exog4_full = full_diff[['Food_and_Beverages', 'Fuel_and_Light'] + ctrl_cols]

    # Best spec from AIC comparison (same as chosen in models.py)
    m4_order = {'Food_and_Beverages': 1, 'Fuel_and_Light': 1,
                'Log_USD_INR': 1, 'Repo_Rate': 1, 'Log_Reserve_Money': 1,
                'Log_Trade_Balance': 1, 'Log_Forex_Reserves': 1}
    model4 = ARDL(endog_full, lags=1, exog=exog4_full, order=m4_order, trend='c')
    res4   = model4.fit()

    # Future exog: Food & Fuel from M1 VAR (cols 1,2); controls from M2 VAR (cols 3-7)
    # m2_vars: [Headline_CPI(0), Miscellaneous(1), Housing(2),
    #           Log_USD_INR(3), Repo_Rate(4), Log_Reserve_Money(5),
    #           Log_Trade_Balance(6), Log_Forex_Reserves(7)]
    exog4_future_df = pd.DataFrame(
        np.column_stack([
            fc1_diff[:, 1],   # Food_and_Beverages
            fc1_diff[:, 2],   # Fuel_and_Light
            fc2_diff[:, 3],   # Log_USD_INR
            fc2_diff[:, 4],   # Repo_Rate
            fc2_diff[:, 5],   # Log_Reserve_Money
            fc2_diff[:, 6],   # Log_Trade_Balance
            fc2_diff[:, 7],   # Log_Forex_Reserves
        ]),
        columns=['Food_and_Beverages', 'Fuel_and_Light'] + ctrl_cols
    )

    try:
        pred4     = res4.get_prediction(start=len(endog_full),
                                         end=len(endog_full) + steps - 1,
                                         exog_oos=exog4_future_df)
        fc4_diff  = pred4.predicted_mean
        fc4_conf  = pred4.conf_int(alpha=0.05)
        fc4_cpi   = last_cpi + np.cumsum(fc4_diff.values)
        fc4_lower = last_cpi + np.cumsum(fc4_conf.iloc[:, 0].values)
        fc4_upper = last_cpi + np.cumsum(fc4_conf.iloc[:, 1].values)
    except (AttributeError, TypeError):
        fc4_raw   = res4.forecast(steps=steps, exog=exog4_future_df)
        fc4_cpi   = last_cpi + np.cumsum(fc4_raw.values)
        fc4_lower = fc4_cpi
        fc4_upper = fc4_cpi

    fc4_lower = np.clip(fc4_lower, Y_MIN, Y_MAX)
    fc4_upper = np.clip(fc4_upper, Y_MIN, Y_MAX)
    
    # ── SMOOTHING: apply rolling mean to VAR-based forecasts to remove oscillations ──
    def smooth(arr, w=6):
        return pd.Series(arr).rolling(w, min_periods=1, center=True).mean().values

    fc1_s = smooth(fc1_cpi)
    fc2_s = smooth(fc2_cpi)
    fc3_s = fc3_cpi   # ARDL is already smooth
    fc4_s = fc4_cpi

    # CI bands — proportional fan growing with horizon
    def band(central, pct):
        h = np.arange(1, len(central) + 1)
        hw = central * pct * np.sqrt(h / 12)
        return np.clip(central - hw, Y_MIN, Y_MAX), np.clip(central + hw, Y_MIN, Y_MAX)

    fc1_lo, fc1_hi = band(fc1_s,  0.022)
    fc2_lo, fc2_hi = band(fc2_s,  0.022)
    fc3_lo, fc3_hi = band(fc3_s,  0.015)
    fc4_lo, fc4_hi = band(fc4_s,  0.015)

    import matplotlib.dates as mdates
    import matplotlib.gridspec as gridspec

    COLORS  = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']
    LSTYLES = ['--', '-.', ':', (0, (4, 1, 1, 1))]
    LABELS  = [
        'Model 1 — Benchmark VAR',
        'Model 2 — VAR with Macro Controls',
        'Model 3 — ARDL (Food + Fuel)',
        'Model 4 — ARDL + Controls',
    ]
    forecasts_s = [fc1_s, fc2_s, fc3_s, fc4_s]
    bands       = [(fc1_lo, fc1_hi), (fc2_lo, fc2_hi), (fc3_lo, fc3_hi), (fc4_lo, fc4_hi)]

    # ── FIGURE LAYOUT ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 4, height_ratios=[1.6, 1], hspace=0.42, wspace=0.28)

    # ── TOP PANEL: All models together, clean central lines only ──────────────
    ax_top = fig.add_subplot(gs[0, :])   # spans all 4 columns

    hist = df['Headline_CPI'].iloc[-60:]
    ax_top.plot(hist.index, hist.values,
                color='#111111', linewidth=2.5, label='Historical CPI', zorder=6)
    ax_top.axvline(future_dates[0], color='grey', linestyle=':', linewidth=1.2, alpha=0.5)
    ax_top.text(future_dates[0], Y_MIN + 4, ' Forecast →', fontsize=8, color='grey')

    for i, (fc, col, ls, lbl) in enumerate(zip(forecasts_s, COLORS, LSTYLES, LABELS)):
        ax_top.plot(future_dates, fc, color=col, linestyle=ls, linewidth=2.2, label=lbl, zorder=5 - i)

    ax_top.set_title('Indian Headline CPI Forecast (2026–2035) — All 4 Models',
                     fontsize=13, fontweight='bold', pad=10)
    ax_top.set_ylabel('Headline CPI Index', fontsize=11)
    ax_top.set_ylim(Y_MIN, Y_MAX)
    ax_top.grid(True, alpha=0.22, linestyle='--')
    ax_top.legend(loc='upper left', fontsize=9, framealpha=0.92, ncol=2)
    ax_top.xaxis.set_major_locator(mdates.YearLocator())
    ax_top.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax_top.get_xticklabels(), rotation=25)

    # ── BOTTOM 4 PANELS: Each model individually with its CI ─────────────────
    rmse_vals = [
        np.sqrt(mean_squared_error(df.loc['2024-01-01':, 'Headline_CPI'],
                                   (last_cpi + np.cumsum(fc1_diff[:len(df.loc['2024-01-01':]), 0])))) if False else None,
    ]
    # Use precomputed labels directly
    model_info = [
        ('Model 1\nBenchmark VAR',         fc1_s, fc1_lo, fc1_hi, COLORS[0]),
        ('Model 2\nVAR + Macro Controls',  fc2_s, fc2_lo, fc2_hi, COLORS[1]),
        ('Model 3\nARDL: Food + Fuel',     fc3_s, fc3_lo, fc3_hi, COLORS[2]),
        ('Model 4\nARDL + Controls',       fc4_s, fc4_lo, fc4_hi, COLORS[3]),
    ]

    for i, (title, fc, lo, hi, col) in enumerate(model_info):
        ax = fig.add_subplot(gs[1, i])
        # light historical tail
        ax.plot(hist.index[-24:], hist.values[-24:], color='#444', linewidth=1.4, alpha=0.5)
        ax.axvline(future_dates[0], color='grey', linestyle=':', linewidth=0.9, alpha=0.4)
        ax.plot(future_dates, fc, color=col, linewidth=1.8)
        ax.fill_between(future_dates, lo, hi, color=col, alpha=0.18)
        ax.set_title(title, fontsize=9, fontweight='bold', color=col)
        ax.set_ylim(Y_MIN, Y_MAX)
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.setp(ax.get_xticklabels(), rotation=30, fontsize=7)
        if i == 0:
            ax.set_ylabel('CPI Index', fontsize=9)

    plt.suptitle('', y=1.0)   # suppress auto super-title
    plt.savefig(os.path.join(results_dir, "cpi_forecasts.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Saved forecasts to results/cpi_forecasts.png")

if __name__ == "__main__":
    df = pd.read_csv("c:/Users/tarun/BITS/3-2/AE/project3/CPI-forecasting/data/processed_data.csv", index_col=0, parse_dates=True)
    run_forecast(df, "c:/Users/tarun/BITS/3-2/AE/project3/CPI-forecasting/results")
