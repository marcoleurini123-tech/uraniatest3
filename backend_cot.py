
import os
import pandas as pd
import numpy as np

CFTC_MAPPING = {
    "E-MINI S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE": "SPX",
    "E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE": "SPX",
    "VIX FUTURES - CBOE FUTURES EXCHANGE": "VIX",
    "NASDAQ-100 STOCK INDEX (MINI) - CHICAGO MERCANTILE EXCHANGE": "Nasdaq",
    "NASDAQ MINI - CHICAGO MERCANTILE EXCHANGE": "Nasdaq",
    "RUSSELL 2000 MINI INDEX FUTURE - ICE FUTURES U.S.": "Russell 2000",
    "RUSSELL E-MINI - CHICAGO MERCANTILE EXCHANGE": "Russell 2000",
    "NIKKEI STOCK AVERAGE - CHICAGO MERCANTILE EXCHANGE": "Nikkei",
    "10-YEAR U.S. TREASURY NOTES - CHICAGO BOARD OF TRADE": "10Y UST",
    "UST 10Y NOTE - CHICAGO BOARD OF TRADE": "10Y UST",
    "2-YEAR U.S. TREASURY NOTES - CHICAGO BOARD OF TRADE": "2Y UST",
    "2Y NOTE - CHICAGO BOARD OF TRADE": "2Y UST",
    "5-YEAR U.S. TREASURY NOTES - CHICAGO BOARD OF TRADE": "5Y UST",
    "U.S. TREASURY BONDS - CHICAGO BOARD OF TRADE": "UST Bonds",
    "EURO FX - CHICAGO MERCANTILE EXCHANGE": "EUR",
    "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE": "JPY",
    "BRITISH POUND STERLING - CHICAGO MERCANTILE EXCHANGE": "GBP",
    "BRITISH POUND - CHICAGO MERCANTILE EXCHANGE": "GBP",
    "AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE": "AUD",
    "CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE": "CAD",
    "SWISS FRANC - CHICAGO MERCANTILE EXCHANGE": "CHF",
    "NEW ZEALAND DOLLAR - CHICAGO MERCANTILE EXCHANGE": "NZD",
    "U.S. DOLLAR INDEX - ICE FUTURES U.S.": "USD Index",
    "USD INDEX - ICE FUTURES U.S.": "USD Index",
    "GOLD - COMMODITY EXCHANGE INC.": "Gold",
    "SILVER - COMMODITY EXCHANGE INC.": "Silver",
    "COPPER-GRADE #1 - COMMODITY EXCHANGE INC.": "Copper",
    "COPPER- #1 - COMMODITY EXCHANGE INC.": "Copper",
    "CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE": "Crude Oil",
    "WTI-PHYSICAL - NEW YORK MERCANTILE EXCHANGE": "Crude Oil",
    "BRENT CRUDE OIL LAST DAY - NEW YORK MERCANTILE EXCHANGE": "Brent Crude",
    "NATURAL GAS - NEW YORK MERCANTILE EXCHANGE": "Nat Gas",
    "CORN - CHICAGO BOARD OF TRADE": "Corn",
    "SOYBEANS - CHICAGO BOARD OF TRADE": "Soybeans",
    "WHEAT-SRW - CHICAGO BOARD OF TRADE": "Wheat",
    "SUGAR NO. 11 - ICE FUTURES U.S.": "Sugar",
    "COFFEE C - ICE FUTURES U.S.": "Coffee",
    "COCOA - ICE FUTURES U.S.": "Cocoa",
    "BITCOIN - CHICAGO MERCANTILE EXCHANGE": "BTC"
}

def fetch_cftc_data(years_back: int = 4) -> pd.DataFrame:
    """
    Legge la matrice COT prioritaria dal database locale 'cot_history.csv' 
    per garantire stabilità e conformità alla Regola 1 senza dipendere da timeout web.
    """
    csv_path = "cot_history.csv"
    
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, low_memory=False)
            if not df.empty and 'Data' in df.columns:
                df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
                df = df.dropna(subset=['Data'])
                if 'Net_NC' not in df.columns and 'NC_Long' in df.columns and 'NC_Short' in df.columns:
                    df['Net_NC'] = df['NC_Long'] - df['NC_Short']
                if 'Net_Comm' not in df.columns and 'Comm_Long' in df.columns and 'Comm_Short' in df.columns:
                    df['Net_Comm'] = df['Comm_Long'] - df['Comm_Short']
                return df
        except Exception as e:
            pass
            
    raise ValueError("IMPOSSIBILE CARICARE LA MATRICE COT: File 'cot_history.csv' assente o non valido nella root.")

def calculate_cot_zscores(df_cot: pd.DataFrame, window_weeks: int = 156) -> pd.DataFrame:
    """
    Calcola lo Z-Score storico sulle Posizioni Nette (Regola 2).
    """
    if df_cot.empty or 'Net_NC' not in df_cot.columns:
        return pd.DataFrame()

    df = df_cot.sort_values(by=['Asset', 'Data']).copy()
    min_obs = 52  
    
    df['NC_Mean'] = df.groupby('Asset')['Net_NC'].transform(lambda x: x.rolling(window_weeks, min_periods=min_obs).mean())
    df['NC_Std'] = df.groupby('Asset')['Net_NC'].transform(lambda x: x.rolling(window_weeks, min_periods=min_obs).std())
    df['Z_NC'] = (df['Net_NC'] - df['NC_Mean']) / (df['NC_Std'] + 1e-9)
    
    if 'Net_Comm' in df.columns:
        df['Comm_Mean'] = df.groupby('Asset')['Net_Comm'].transform(lambda x: x.rolling(window_weeks, min_periods=min_obs).mean())
        df['Comm_Std'] = df.groupby('Asset')['Net_Comm'].transform(lambda x: x.rolling(window_weeks, min_periods=min_obs).std())
        df['Z_Comm'] = (df['Net_Comm'] - df['Comm_Mean']) / (df['Comm_Std'] + 1e-9)
    else:
        df['Z_Comm'] = 0.0
        
    latest = df.groupby('Asset').tail(1).copy()
    latest = latest.dropna(subset=['Z_NC'])
    
    latest['Z_NC'] = latest['Z_NC'].round(2)
    latest['Z_Comm'] = latest['Z_Comm'].round(2)
    
    latest['Alert_NC'] = np.where(latest['Z_NC'].abs() >= 1.8, "⭐", "")
    latest['Alert_Comm'] = np.where(latest['Z_Comm'].abs() >= 1.8, "⭐", "") if 'Z_Comm' in latest.columns else ""
    
    cols_to_return = [col for col in ['Data', 'Asset', 'Net_NC', 'Z_NC', 'Alert_NC', 'Net_Comm', 'Z_Comm', 'Alert_Comm'] if col in latest.columns]
    
    sort_col = 'Z_Comm' if 'Z_Comm' in latest.columns else 'Z_NC'
    latest = latest[cols_to_return].sort_values(by=sort_col, key=abs, ascending=False).reset_index(drop=True)
    
    return latest
