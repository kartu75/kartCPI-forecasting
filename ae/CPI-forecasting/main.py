import os
import sys

def main():
    print("="*50)
    print(" Forecasting Indian Headline CPI - Main Pipeline ")
    print("="*50)
    
    # Define directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(base_dir, "src")
    sys.path.append(src_dir)
    
    raw_dir = os.path.join(base_dir, "data", "raw")
    processed_file = os.path.join(base_dir, "data", "processed_data.csv")
    results_dir = os.path.join(base_dir, "results")
    
    # 1. Data Preparation
    print("\n[1/5] Running Data Preparation...")
    import data_prep
    df = data_prep.load_and_prep_data(raw_dir)
    df.to_csv(processed_file)
    print(f"Data shape: {df.shape}. Saved to {processed_file}")
    
    # 2. Exploratory Data Analysis
    print("\n[2/5] Running Exploratory Data Analysis (EDA)...")
    import eda
    eda.run_eda(df, results_dir)
    
    # 3. Stationarity & Cointegration Tests
    print("\n[3/5] Running Stationarity & Cointegration Tests...")
    import tests
    tests.run_tests(df, results_dir)
    
    # 4. Econometric Modeling & Evaluation
    print("\n[4/5] Running Econometric Models (VAR, ARDL, ARDL+Controls)...")
    import models
    metrics = models.evaluate_models(df, results_dir)
    
    # 5. Out-of-Sample Forecasting
    print("\n[5/5] Generating Out-of-Sample Forecasts (up to 2035)...")
    import forecast
    forecast.run_forecast(df, results_dir, steps=120)
    
    print("\n" + "="*50)
    print(" Pipeline completed successfully! Check the 'results/' folder. ")
    print("="*50)

if __name__ == "__main__":
    main()
