# Forecasting Indian Headline CPI: Services, Housing, and Commodity Shocks vs. Traditional Models

This project develops an applied econometric pipeline to forecast the Indian Headline Consumer Price Index (CPI). It evaluates traditional benchmark models against proposed models that incorporate services, housing, and external macroeconomic shocks (proxied by the USD/INR exchange rate, repo rate, money supply, trade balance, and forex reserves).

## Project Structure
- `data/raw/`: Directory containing raw `.csv` and `.xlsx` datasets from the RBI DBIE and other sources.
- `data/processed_data.csv`: The combined, aligned, and interpolated monthly panel data from Jan 2015 to Dec 2025.
- `src/`: Contains all the Python modules for the MVP pipeline.
  - `data_prep.py`: Cleans and merges multiple Excel/CSV files into a unified monthly dataset.
  - `eda.py`: Performs Exploratory Data Analysis, generating line plots and ACF/PACF charts.
  - `tests.py`: Conducts Augmented Dickey-Fuller (ADF) tests for unit roots and Johansen tests for cointegration.
  - `models.py`: Fits and evaluates the four core econometric models on a Train/Test split.
  - `forecast.py`: Uses the full dataset to generate out-of-sample forecasts up to 2035 with 95% confidence intervals.
- `results/`: Directory containing generated plots, summaries, and textual reports (stationarity, cointegration, model evaluation).
- `main.py`: The master execution script that runs the entire pipeline end-to-end.

## Dependencies
- `pandas` (Data manipulation)
- `numpy` (Numerical computing)
- `statsmodels` (Econometric modeling, VAR, VECM, ADF, ARDL, ARMAX)
- `matplotlib` (Data visualization)
- `scikit-learn` (Metrics calculation like RMSE, MAE)
- `openpyxl`, `xlrd` (Excel parsing)

### Setup Virtual Environment
It is recommended to use a Python virtual environment to manage dependencies cleanly.

**1. Create the virtual environment:**
```bash
python -m venv venv
```

**2. Activate the virtual environment:**
- On **Windows**:
  ```bash
  venv\Scripts\activate
  ```
- On **macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
# or manually:
pip install pandas numpy statsmodels arch matplotlib scikit-learn openpyxl xlrd
```

## How to Run the Analysis

### Option 1: Automated Pipeline
To execute the complete pipeline automatically, simply run the master script from the root directory:
```bash
python main.py
```
This will sequentially read the raw data, process it, evaluate the models, generate forecasts, and output all reports and plots to the `results/` folder.

### Option 2: Step-by-Step Manual Execution
If you prefer to review the outputs of each stage individually or debug specific sections, you can execute the modules sequentially from the `src/` directory:

1. **Data Preparation**: Clean and align all raw `.xlsx` and `.csv` files.
   ```bash
   python src/data_prep.py
   ```
   *Output: `data/processed_data.csv`*

2. **Exploratory Data Analysis (EDA)**: Generate summary statistics, line plots, and ACF/PACF charts.
   ```bash
   python src/eda.py
   ```
   *Outputs: `results/summary_statistics.csv`, `results/variable_line_plots.png`, `results/cpi_acf_pacf.png`*

3. **Stationarity & Cointegration Tests**: Run ADF and Johansen tests.
   ```bash
   python src/tests.py
   ```
   *Output: `results/stationarity_cointegration_report.txt`*

4. **Model Evaluation**: Train the VAR, ARDL, and ARDL+Controls models on a split dataset and calculate RMSE/MAE.
   ```bash
   python src/models.py
   ```
   *Output: `results/model_evaluation_report.txt`*

5. **Forecasting**: Train on the full dataset and generate 10-year forecasts with 95% confidence intervals.
   ```bash
   python src/forecast.py
   ```
   *Output: `results/cpi_forecasts.png`*

## Econometric Theory

### Stationarity and Cointegration
Macroeconomic variables are rarely stationary. We use the **Augmented Dickey-Fuller (ADF) Test** to check for unit roots. If variables are $I(1)$ (stationary only after first differencing), we test for long-run relationships using the **Johansen Cointegration Test**. The presence of cointegration suggests that a Vector Error Correction Model (VECM) or VAR in levels/differences can be used to capture both short-term dynamics and long-term equilibrium.

### Model 1: Benchmark OLS (Single-Equation)
The traditional approach models CPI based primarily on volatile components like Food and Fuel.
We estimate a **single-equation OLS regression** where the change in Headline CPI is the dependent variable and the **lag-1 levels** of CPI, Food, and Fuel are the independent variables.

**Formula:**
$$
\Delta \text{CPI}_t = c
  + \alpha_1\, \text{CPI}_{t-1}
  + \beta_1\, \text{Food}_t
  + \gamma_1\, \text{Fuel}_t
  + u_t
$$

Where:
- $\Delta \text{CPI}_t = \text{CPI}_t - \text{CPI}_{t-1}$ is the **dependent variable** (month-on-month change in Headline CPI)
- $\text{CPI}_{t-1}$ is the **only lagged variable** (own lag of CPI)
- $\text{Food}_t$, $\text{Fuel}_t$ are **contemporaneous independent variables** (Food and Beverages, Fuel and Light at time $t$)
- $c$ is a constant; $u_t$ is the error term

### Model 2: Proposed OLS (Single-Equation, CPI in Levels)
Modern inflation is increasingly driven by sticky components (Housing, Services) and external shocks. We estimate a **single-equation OLS regression** where the **level of Headline CPI** is the dependent variable and the **lag-1 level** of CPI and the **contemporaneous values** of all other predictors are the independent variables:
- `CPI_{t-1}` (own lag — only lagged variable)
- `Food_and_Beverages_t`, `Fuel_and_Light_t` (supply-side shocks)
- `Miscellaneous_t` (proxy for Services)
- `Housing_t`
- `Log_USD_INR_t`
- `Repo_Rate_t`, `Log_Reserve_Money_t`, `Log_Trade_Balance_t`, `Log_Forex_Reserves_t`

**Formula:**
$$
\text{CPI}_t = \mu
  + \phi_1\, \text{CPI}_{t-1}
  + \alpha_1\, \text{Food}_t
  + \alpha_2\, \text{Fuel}_t
  + \beta_1\, \text{Misc}_t
  + \beta_2\, \text{Housing}_t
  + \beta_3\, \log(\text{USD/INR})_t
  + \beta_4\, \text{RepoRate}_t
  + \beta_5\, \log(\text{ReserveM})_t
  + \beta_6\, \log(\text{TradeBal})_t
  + \beta_7\, \log(\text{Forex})_t
  + \varepsilon_t
$$

Where:
- $\text{CPI}_t$ is the **dependent variable** (level of Headline CPI)
- $\text{CPI}_{t-1}$ is the **only lagged variable** (own lag)
- All other right-hand-side terms are **contemporaneous independent variables** at time $t$
- $\mu$ is a constant; $\varepsilon_t$ is the error term

*Note: Since CPI is persistent (near unit root), including $\text{CPI}_{t-1}$ as a regressor makes this an autoregressive (AR(1)-augmented) specification in levels, capturing the strong momentum in CPI while allowing all macro controls to influence the level jointly.*

### Model 3: Autoregressive Distributed Lag — ARDL(1, 1, 1)

To quantify the specific distributed-lag impacts of food and fuel price shocks on Headline CPI, we employ an **Autoregressive Distributed Lag (ARDL)** model. Unlike the full VAR system (Models 1 & 2), ARDL focuses on a single equation for `Δ Headline CPI`, conditioning on its own past values and the current and lagged changes in the two key drivers from Model 1: `Food_and_Beverages` and `Fuel_and_Light`.

#### Lag-Order Selection

Three candidate specifications were evaluated on the **training set (2015–2023)** using **AIC** (primary criterion) and **BIC** (secondary), with out-of-sample RMSE and MAE computed on the **test set (2024–2025)**. The AR lag is **fixed at 1** across all candidates; only the distributed lags for Food and Fuel are varied:

| Specification  | AIC      | BIC      | RMSE   | MAE    |
|----------------|----------|----------|--------|--------|
| **ARDL(1, 1, 1)**  | **−53.32** | **−34.68** | **0.3647** | **0.2738** |
| ARDL(1, 2, 2)  | −52.52   | −28.55   | 0.4213 | 0.3213 |
| ARDL(1, 3, 3)  | −51.40   | −22.10   | 0.4798 | 0.3632 |

*(Values populated after running the pipeline)*

**Winner: ARDL(1, 1, 1)** — Achieves the lowest AIC among the three candidates. With the AR lag fixed at 1, adding more distributed lags for Food and Fuel worsens both AIC and out-of-sample RMSE, confirming that a single contemporaneous + one-lag structure is sufficient.

#### Model Specification

The selected model, **ARDL(1, 1, 1)**, is defined on first-differenced data as:

$$
\Delta \text{CPI}_t = c
  + \phi_1\, \Delta \text{CPI}_{t-1}
  + \beta_{10}\, \Delta \text{Food}_t + \beta_{11}\, \Delta \text{Food}_{t-1}
  + \gamma_{10}\, \Delta \text{Fuel}_t + \gamma_{11}\, \Delta \text{Fuel}_{t-1}
  + \epsilon_t
$$

Where:
- $p = 1$: one autoregressive lag of the dependent variable $\Delta\text{CPI}$
- $q_1 = 1$: one distributed lag for $\Delta\text{Food\_and\_Beverages}$ (current + one lag)
- $q_2 = 1$: one distributed lag for $\Delta\text{Fuel\_and\_Light}$ (current + one lag)
- $c$ is a constant term; $\epsilon_t \sim \text{WN}(0, \sigma^2)$

---

### Model 4: ARDL with Macro Controls — ARDL4(1, 1, 1ctrl)

Model 4 extends Model 3 by adding **five macro control variables** as additional exogenous regressors, capturing the external policy and financial environment that Model 3 ignores. The controls are borrowed from the Model 2 VAR: `Log_USD_INR`, `Repo_Rate`, `Log_Reserve_Money`, `Log_Trade_Balance`, and `Log_Forex_Reserves`.

This allows the ARDL framework to attribute short-run CPI dynamics to both the direct supply-side shocks (food and fuel) and the broader monetary and external sector conditions simultaneously.

#### Lag-Order Selection

Three candidate specifications were evaluated on the **training set (2015–2023)** using **AIC** (primary) and **BIC** (secondary). Food/fuel lags are kept at 1 (t and t-1); control variable lags are tested at 0 (only contemporaneous) vs 1 (t and t-1):

| Specification            | AIC      | BIC    | RMSE   | MAE    |
|--------------------------|----------|--------|--------|--------|
| **ARDL4(1, 1, 0ctrl)** | **−49.39** | **−17.43** | 0.6112 | 0.4506 |
| ARDL4(1, 1, 1ctrl) | −44.51 | 0.77 | **0.6064** | **0.4471** |

**Winner: ARDL4(1, 1, 0ctrl)** — Achieves the lowest AIC. While the out-of-sample RMSE is very slightly higher for the 0-lag controls, adding 5 extra lagged parameter terms heavily penalizes the AIC and BIC. Therefore, using the contemporaneous control variables only (0 lag) provides the best balance of model fit and parsimony.

#### Model Specification

The selected model, **ARDL4(1, 1, 0ctrl)**, is defined on first-differenced data as:

$$
\Delta \text{CPI}_t = c
  + \phi_1\, \Delta \text{CPI}_{t-1}
  + \beta_{10}\, \Delta \text{Food}_t + \beta_{11}\, \Delta \text{Food}_{t-1}
  + \gamma_{10}\, \Delta \text{Fuel}_t + \gamma_{11}\, \Delta \text{Fuel}_{t-1}
  + \delta_1\, \Delta \log(\text{USD/INR})_t
  + \lambda_1\, \Delta \text{RepoRate}_t
  + \mu_1\, \Delta \log(\text{ReserveM})_t
  + \nu_1\, \Delta \log(\text{TradeBal})_t
  + \rho_1\, \Delta \log(\text{Forex})_t
  + \epsilon_t
$$

Where:
- $p = 1$: one AR lag for $\Delta\text{CPI}$
- $q_{\text{food}} = q_{\text{fuel}} = 1$: one distributed lag for food and fuel shocks
- $q_{\text{ctrl}} = 0$: only contemporaneous terms for the five macro controls
- $c$ is a constant term; $\epsilon_t \sim \text{WN}(0, \sigma^2)$

**For out-of-sample forecasting**, the future values of the macro controls are sourced from the Model 2 VAR projections, ensuring full consistency across the pipeline.
