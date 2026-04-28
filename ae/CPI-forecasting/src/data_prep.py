import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

def load_and_prep_data(raw_dir):
    # 1. Process CPI
    cpi_file = os.path.join(raw_dir, "CPI - Rural, Urban, Combined.xlsx")
    df_cpi_raw = pd.read_excel(cpi_file, sheet_name=0, header=None)
    
    # Extract relevant columns
    # In the provided data, Month is col 1, Description is col 2, Combined Index is col 8
    df_cpi = df_cpi_raw.iloc[7:, [1, 2, 8]].copy()
    df_cpi.columns = ['Date', 'Commodity', 'Index']
    df_cpi.dropna(subset=['Date', 'Commodity', 'Index'], inplace=True)
    
    # Filter the required commodities
    commodities_map = {
        'A) General Index': 'Headline_CPI',
        'A.1) Food and beverages': 'Food_and_Beverages',
        'A.5) Fuel and light': 'Fuel_and_Light',
        'A.4) Housing': 'Housing',
        'A.6) Miscellaneous': 'Miscellaneous'
    }
    df_cpi = df_cpi[df_cpi['Commodity'].isin(commodities_map.keys())]
    df_cpi['Commodity'] = df_cpi['Commodity'].map(commodities_map)
    df_cpi['Index'] = pd.to_numeric(df_cpi['Index'], errors='coerce')
    
    # Pivot to get commodities as columns
    df_cpi = df_cpi.pivot_table(index='Date', columns='Commodity', values='Index', aggfunc='mean')
    df_cpi.index = pd.to_datetime(df_cpi.index, format='%b-%Y')
    df_cpi = df_cpi.resample('MS').mean()
    
    # 2. Process USD_INR (source: DEXINUS.xlsx — FRED daily INR per USD)
    usd_file = os.path.join(raw_dir, "DEXINUS.xlsx")
    df_usd_raw = pd.read_excel(usd_file, sheet_name='Daily')
    df_usd_raw['observation_date'] = pd.to_datetime(df_usd_raw['observation_date'])
    df_usd_raw.set_index('observation_date', inplace=True)
    df_usd_raw['DEXINUS'] = pd.to_numeric(df_usd_raw['DEXINUS'], errors='coerce')
    df_usd_raw.dropna(subset=['DEXINUS'], inplace=True)   # drop holiday NaNs
    df_usd = df_usd_raw[['DEXINUS']].resample('MS').mean()
    df_usd.rename(columns={'DEXINUS': 'USD_INR'}, inplace=True)
    
    # Handle missing USD_INR data after March 2020 with a custom trajectory
    last_date = df_usd.index[-1]
    target_end = pd.to_datetime('2025-12-01')
    if last_date < target_end:
        # User requested: peak at 91, fall to 89 by Dec 2025, smooth semi-linear trend.
        # We will fit a parabolic curve: y = a(x - x_p)^2 + 91
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), end=target_end, freq='MS')
        num_months = len(future_dates)
        current_val = df_usd['USD_INR'].iloc[-1]
        
        # We know y(0) = current_val, y(num_months) = 89. Peak is 91 at x_p.
        # (num_months - x_p) / x_p = sqrt( (91 - 89) / (91 - current_val) )
        ratio = np.sqrt(2.0 / (91.0 - current_val))
        x_p = num_months / (1.0 + ratio)
        a = (current_val - 91.0) / (x_p**2)
        
        future_vals = []
        for i in range(1, num_months + 1):
            val = a * (i - x_p)**2 + 91.0
            future_vals.append(val)
            
        df_future = pd.DataFrame({'USD_INR': future_vals}, index=future_dates)
        df_usd = pd.concat([df_usd, df_future])
        
    df_usd['Log_USD_INR'] = np.log(df_usd['USD_INR'])
    
    # 3. Process Repo Rate
    repo_file = os.path.join(raw_dir, "Major Monetary Policy Rates and Reserve Requirements - Bank Rate, LAF (Repo, Reverse Repo, SDF and MSF) Rates, CRR & SLR.xlsx")
    df_repo_raw = pd.read_excel(repo_file, sheet_name=0, header=None)
    # Effective Date is col 1, Repo is col 3
    df_repo = df_repo_raw.iloc[8:, [1, 3]].copy()
    df_repo.columns = ['Date', 'Repo_Rate']
    df_repo.dropna(inplace=True)
    df_repo['Date'] = pd.to_datetime(df_repo['Date'], errors='coerce')
    df_repo['Repo_Rate'] = pd.to_numeric(df_repo['Repo_Rate'], errors='coerce')
    df_repo.dropna(inplace=True)
    df_repo.set_index('Date', inplace=True)
    df_repo = df_repo.resample('MS').mean().ffill() # Forward fill as rates stay constant until changed
    
    # 4. Reserve Money
    money_file = os.path.join(raw_dir, "RBIB Table No. 11 _ Reserve Money_ Components and Sources.xlsx")
    df_money_raw = pd.read_excel(money_file, sheet_name=0, header=None)
    # Assume Date is col 1, Reserve Money is col 2 (we will see if this fails and fix it)
    try:
        df_money = df_money_raw.iloc[5:, [1, 2]].copy()
        df_money.columns = ['Date', 'Reserve_Money']
        df_money['Date'] = pd.to_datetime(df_money['Date'], errors='coerce')
        df_money['Reserve_Money'] = pd.to_numeric(df_money['Reserve_Money'], errors='coerce')
        df_money.dropna(inplace=True)
        df_money.set_index('Date', inplace=True)
        df_money = df_money.resample('MS').mean()
    except Exception as e:
        print(f"Error parsing Reserve Money: {e}")
        df_money = pd.DataFrame()
        
    # 5. Forex Reserves
    forex_file = os.path.join(raw_dir, "RBIB Table No. 33 _ Foreign Exchange Reserves - Weekly.xlsx")
    df_forex_raw = pd.read_excel(forex_file, sheet_name=0, header=None)
    # Assume Date is col 1, Total Reserves is col 2
    try:
        df_forex = df_forex_raw.iloc[5:, [1, 2]].copy()
        df_forex.columns = ['Date', 'Forex_Reserves']
        df_forex['Date'] = pd.to_datetime(df_forex['Date'], errors='coerce')
        df_forex['Forex_Reserves'] = pd.to_numeric(df_forex['Forex_Reserves'], errors='coerce')
        df_forex.dropna(inplace=True)
        df_forex.set_index('Date', inplace=True)
        df_forex = df_forex.resample('MS').mean()
    except Exception as e:
        print(f"Error parsing Forex: {e}")
        df_forex = pd.DataFrame()
        
    # 6. Trade Balance
    trade_file = os.path.join(raw_dir, "India’s Foreign Trade - US Dollar.xlsx")
    df_trade_raw = pd.read_excel(trade_file, sheet_name=0, header=None)
    try:
        # col 1=Year, col 2=Month, col 3=Exports, col 4=Imports
        df_trade = df_trade_raw.iloc[7:, [1, 2, 3, 4]].copy()
        df_trade.columns = ['Year', 'Month', 'Exports', 'Imports']
        df_trade.dropna(inplace=True)
        
        def parse_trade_date(row):
            y_str = str(row['Year']).strip()
            if len(y_str) >= 4:
                base_year = int(y_str[:4])
                m_str = str(row['Month']).strip()
                if m_str in ['January', 'February', 'March']:
                    return pd.to_datetime(f"{base_year+1}-{m_str}-01")
                else:
                    return pd.to_datetime(f"{base_year}-{m_str}-01")
            return pd.NaT
            
        df_trade['Date'] = df_trade.apply(parse_trade_date, axis=1)
        for col in ['Exports', 'Imports']:
            df_trade[col] = pd.to_numeric(df_trade[col], errors='coerce')
            
        df_trade.dropna(subset=['Date', 'Exports', 'Imports'], inplace=True)
        df_trade['Log_Trade_Balance'] = np.log(df_trade['Exports']) - np.log(df_trade['Imports'])
        df_trade.set_index('Date', inplace=True)
        df_trade = df_trade[['Log_Trade_Balance']].resample('MS').mean()
    except Exception as e:
        print(f"Error parsing Trade Balance: {e}")
        df_trade = pd.DataFrame()

    # Merge all DataFrames
    dfs = [df_cpi, df_usd, df_repo, df_money, df_forex, df_trade]
    df_final = dfs[0]
    for i, df in enumerate(dfs[1:]):
        if not df.empty:
            df_final = df_final.merge(df, left_index=True, right_index=True, how='outer')
            
    # Filter timeframe Jan 2015 to Dec 2025
    df_final = df_final.loc['2015-01-01':'2025-12-01']
    
    # Handle missing values: Interpolate linearly for up to 3 months, then ffill/bfill
    df_final.interpolate(method='linear', limit=3, inplace=True)
    df_final.ffill(inplace=True)
    df_final.bfill(inplace=True)
    
    # Calculate Log transformations
    df_final['Log_Reserve_Money'] = np.log(df_final['Reserve_Money'])
    df_final['Log_Forex_Reserves'] = np.log(df_final['Forex_Reserves'])
    # Trade Balance is log ratio of Exports to Imports (approximates log(Exports) - log(Imports))
    # df_trade already has Log_Trade_Balance if we calculate it there, but since we have Trade_Balance right now... 
    # Wait, we need to fix df_trade parsing if we want Log(Exports) - Log(Imports).
    # Since we only have Trade_Balance in df_final right now, let's fix df_trade parsing directly.
    
    return df_final

if __name__ == "__main__":
    raw_dir = r"c:\Users\tarun\BITS\3-2\AE\project3\CPI-forecasting\data\raw"
    df = load_and_prep_data(raw_dir)
    print(df.head())
    print(df.info())
    
    out_path = "c:/Users/tarun/BITS/3-2/AE/project3/CPI-forecasting/data/processed_data.csv"
    try:
        df.to_csv(out_path)
        print(f"Saved processed data to {out_path}")
    except PermissionError:
        print("="*60)
        print(" ERROR: PERMISSION DENIED ".center(60, "="))
        print(f"Cannot save to {out_path}.")
        print("The file is currently OPEN and LOCKED by another program.")
        print("-> PLEASE CLOSE EXCEL OR ANY OTHER PROGRAM USING THIS FILE <-")
        print("="*60)
