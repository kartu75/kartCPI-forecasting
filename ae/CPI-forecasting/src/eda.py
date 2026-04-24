import pandas as pd
import matplotlib.pyplot as plt
import os
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

def run_eda(df, results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Summary Statistics
    desc = df.describe()
    desc.to_csv(os.path.join(results_dir, "summary_statistics.csv"))
    print("Saved summary statistics to results/summary_statistics.csv")
    
    # 2. Line Plots for all variables
    # We will plot them in subplots
    num_vars = len(df.columns)
    fig, axes = plt.subplots(nrows=(num_vars + 1) // 2, ncols=2, figsize=(15, 3 * ((num_vars + 1) // 2)))
    axes = axes.flatten()
    
    for i, col in enumerate(df.columns):
        axes[i].plot(df.index, df[col], label=col, color='tab:blue')
        axes[i].set_title(col)
        axes[i].grid(True)
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "variable_line_plots.png"), dpi=300)
    plt.close()
    print("Saved line plots to results/variable_line_plots.png")
    
    # 3. ACF and PACF for Target Variable (Headline CPI)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    plot_acf(df['Headline_CPI'].dropna(), lags=40, ax=axes[0])
    plot_pacf(df['Headline_CPI'].dropna(), lags=40, ax=axes[1], method='ywm')
    axes[0].set_title("ACF of Headline CPI")
    axes[1].set_title("PACF of Headline CPI")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "cpi_acf_pacf.png"), dpi=300)
    plt.close()
    print("Saved ACF/PACF plots to results/cpi_acf_pacf.png")

if __name__ == "__main__":
    df = pd.read_csv("c:/Users/tarun/BITS/3-2/AE/project3/CPI-forecasting/data/processed_data.csv", index_col=0, parse_dates=True)
    run_eda(df, "c:/Users/tarun/BITS/3-2/AE/project3/CPI-forecasting/results")
