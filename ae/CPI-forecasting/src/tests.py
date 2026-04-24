import pandas as pd
import numpy as np
import os
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.ardl import ARDL
import statsmodels.api as sm
from scipy.stats import chi2
import warnings
warnings.filterwarnings('ignore')


def run_tests(df, results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    report_lines = []

    # ── ADF UNIT ROOT TESTS ──────────────────────────────────────────────────
    report_lines.append("=== AUGMENTED DICKEY-FULLER (ADF) TEST ===")
    report_lines.append("Null Hypothesis: The series has a unit root (is non-stationary)\n")

    stationary_vars = []
    non_stationary_vars = []

    for col in df.columns:
        result = adfuller(df[col].dropna())
        pval = result[1]
        report_lines.append(f"{col}:")
        report_lines.append(f"  ADF Statistic: {result[0]:.4f}")
        report_lines.append(f"  p-value: {pval:.4f}")

        if pval < 0.05:
            report_lines.append("  Result: Stationary (Reject Null)")
            stationary_vars.append(col)
        else:
            report_lines.append("  Result: Non-Stationary (Fail to Reject Null)")
            non_stationary_vars.append(col)
            diff_result = adfuller(df[col].diff().dropna())
            diff_pval = diff_result[1]
            if diff_pval < 0.05:
                report_lines.append(f"  1st Diff Result: Stationary (p-value: {diff_pval:.4f}) -> I(1)")
            else:
                report_lines.append(f"  1st Diff Result: Non-Stationary (p-value: {diff_pval:.4f}) -> I(>1)")
        report_lines.append("")

    # ── JOHANSEN COINTEGRATION TEST ──────────────────────────────────────────
    report_lines.append("\n=== JOHANSEN COINTEGRATION TEST ===")
    report_lines.append("Testing cointegration among Model 2 variables: Headline_CPI, Miscellaneous, Housing, Log_USD_INR")

    model2_vars = ['Headline_CPI', 'Miscellaneous', 'Housing', 'Log_USD_INR']
    df_johansen = df[model2_vars].dropna()
    johansen_res = coint_johansen(df_johansen, det_order=0, k_ar_diff=1)

    report_lines.append("\nTrace Statistics:")
    for i, trace_stat in enumerate(johansen_res.lr1):
        crit_val = johansen_res.cvt[i, 1]
        report_lines.append(f"r <= {i}: Trace Stat = {trace_stat:.4f}, 5% Critical Value = {crit_val:.4f}")
        if trace_stat > crit_val:
            report_lines.append(f"  -> Reject null of r <= {i}. There is cointegration.")
        else:
            report_lines.append(f"  -> Fail to reject null of r <= {i}. No further cointegration.")

    # ── GRANGER CAUSALITY TESTS ──────────────────────────────────────────────
    diff_df = df.diff().dropna()

    report_lines.append("\n\n=== GRANGER CAUSALITY TESTS ===")
    report_lines.append("H0: The candidate variable does NOT Granger-cause Headline_CPI")
    report_lines.append("Testing on first-differenced data | Max lags = 3 | Significance level = 5%\n")

    granger_candidates = [
        'Food_and_Beverages',
        'Fuel_and_Light',
        'Log_USD_INR',
        'Repo_Rate',
        'Log_Reserve_Money',
        'Log_Trade_Balance',
        'Log_Forex_Reserves',
        'Miscellaneous',
        'Housing',
    ]

    for cand in granger_candidates:
        data_gc = diff_df[['Headline_CPI', cand]].dropna()
        report_lines.append(f"  {cand} -> Headline_CPI:")
        try:
            gc_res = grangercausalitytests(data_gc, maxlag=3, verbose=False)
            for lag in [1, 2, 3]:
                pval  = gc_res[lag][0]['ssr_ftest'][1]
                fstat = gc_res[lag][0]['ssr_ftest'][0]
                sig   = "Significant (Granger-causes CPI)" if pval < 0.05 else "Not significant"
                report_lines.append(
                    f"    Lag {lag}: F-stat = {fstat:.4f}, p-value = {pval:.4f}  [{sig}]"
                )
        except Exception as e:
            report_lines.append(f"    Error: {e}")
        report_lines.append("")

    # ── BREUSCH-GODFREY LM TEST (manual chi-sq version) ─────────────────────
    report_lines.append("\n=== BREUSCH-GODFREY LM TEST (Serial Correlation in Residuals) ===")
    report_lines.append("H0: No serial correlation in ARDL model residuals | Lags tested: 1, 2, 3\n")

    endog   = diff_df['Headline_CPI']
    exog3   = diff_df[['Food_and_Beverages', 'Fuel_and_Light']]
    ctrl_cols = ['Log_USD_INR', 'Repo_Rate', 'Log_Reserve_Money', 'Log_Trade_Balance', 'Log_Forex_Reserves']
    exog4   = diff_df[['Food_and_Beverages', 'Fuel_and_Light'] + ctrl_cols]

    lm_specs = [
        ("Model 3: ARDL(1,1,1)",
         ARDL(endog, lags=1, exog=exog3,
              order={"Food_and_Beverages": 1, "Fuel_and_Light": 1}, trend='c')),
        ("Model 4: ARDL4(1,1,1ctrl)",
         ARDL(endog, lags=1, exog=exog4,
              order={c: 1 for c in ['Food_and_Beverages', 'Fuel_and_Light'] + ctrl_cols}, trend='c')),
    ]

    for name, mdl in lm_specs:
        fitted  = mdl.fit()
        resid   = fitted.resid.dropna().values
        n       = len(resid)
        report_lines.append(f"  {name}:")
        for nlags in [1, 2, 3]:
            # Regress residuals on their lagged values (manual BG chi-sq form)
            y_r = resid[nlags:]
            X_r = np.column_stack([resid[nlags - k: n - k] for k in range(1, nlags + 1)])
            X_r = sm.add_constant(X_r)
            ols_r   = sm.OLS(y_r, X_r).fit()
            lm_stat = len(y_r) * ols_r.rsquared
            lm_pval = 1 - chi2.cdf(lm_stat, df=nlags)
            sig     = "Serial correlation present" if lm_pval < 0.05 else "No serial correlation"
            report_lines.append(
                f"    Lag {nlags}: LM chi2-stat = {lm_stat:.4f}, p-value = {lm_pval:.4f}  [{sig}]"
            )
        report_lines.append("")

    with open(os.path.join(results_dir, "stationarity_cointegration_report.txt"), "w") as f:
        f.write("\n".join(report_lines))

    print("Saved test reports to results/stationarity_cointegration_report.txt")


if __name__ == "__main__":
    df = pd.read_csv("c:/Users/karth/OneDrive/Desktop/ae/CPI-forecasting/data/processed_data.csv",
                     index_col=0, parse_dates=True)
    run_tests(df, "c:/Users/karth/OneDrive/Desktop/ae/CPI-forecasting/results")
