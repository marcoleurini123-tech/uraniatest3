import pandas as pd
import numpy as np
import requests
import io
import yfinance as yf
import os
import math
from datetime import datetime, timedelta
import pandas_datareader.data as web

DB_FILE = "macro_database.csv"
GOOGLE_BRIDGE_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSeeY57SBwd6BftA2Bq8C0nyzzT3wj9WRWOihDF7QE-COPXhC4r2RN_k_BRgZke1nU2BbKT8oRlsXOX/pub?gid=1412711569&single=true&output=csv"

# Definizione rigida delle matrici ammesse nel Database
COLUMNS = [
    "Data", "VIX1D", "VIX9D", "VIX", "VIX3M", "VIX6M", "VIX1Y", "VVIX", "MOVE", "SKEW", 
    "DXY", "DIX", "GEX", "SPY", "RSP", "HYG", "XLY", "XLP", "TLT", "P_C", "GLD", "USO", 
    "Net_Liquidity", "M2"
]

# --- COSTANTI MODULO A (REGIMI MACRO) ---
REGIME_BASKETS = {
    "GOLDILOCKS ECONOMY": ["QQQ", "XLK", "XLY", "IEF", "SMH"],
    "RECESSION": ["TLT", "SHY", "XLU", "XLP", "GLD"],
    "STAGFLATION": ["GLD", "DBC", "XLE", "TIP"],
    "REFLATION": ["XLI", "XLF", "IWM", "EEM", "DBC"],
    "DISINFLATION/SOFT LANDING": ["TLT", "LQD", "QQQ", "VTI", "GLD"],
    "DOLLAR WEAKNESS/GLOBAL REBALANCING": ["EEM", "FXF", "GLD", "IXUS", "DBC"],
    "DEFLATION": ["TLT", "BIL", "SHY", "XLP", "XLU"],
    "DOLLAR WEAKNESS/GLOBAL REBALANCING + BITCOIN": ["EEM", "FXF", "GLD", "IXUS", "IBIT"],
    "DEBASEMENT AGGRESSIVO": ["GLD", "XME", "COPX", "EEM", "IBIT"],
    "DEBASEMENT (SENZA BITCOIN)": ["GLD", "XME", "COPX", "EEM", "VGSH"],
}

TIMEFRAMES = {"Δ 1W": 5, "Δ 1M": 21, "Δ 3M": 63}

# --- FUNZIONI DI BASE E FETCHING DATI ---
def load_db():
    if os.path.exists(DB_FILE):
        df = pd.read_csv(DB_FILE)
        df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
        if df['Data'].dt.tz is not None:
            df['Data'] = df['Data'].dt.tz_localize(None)
        df['Data'] = df['Data'].dt.normalize()
        
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
                
        num_cols = [c for c in COLUMNS if c != "Data"]
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
                
        return df.dropna(subset=['Data']).sort_values("Data")
    return pd.DataFrame(columns=COLUMNS)

def save_db(df):
    df = df.drop_duplicates(subset=['Data'], keep='last').sort_values("Data")
    df.to_csv(DB_FILE, index=False)

def fetch_yahoo_data(days=365):
    tickers_map = {
        "^VIX1D": "VIX1D", "^VIX9D": "VIX9D", "^VIX": "VIX", "^VIX3M": "VIX3M", 
        "^VIX6M": "VIX6M", "^VIX1Y": "VIX1Y", "^VVIX": "VVIX", "^SKEW": "SKEW", 
        "DX-Y.NYB": "DXY", "SPY": "SPY", "RSP": "RSP", "XLY": "XLY", "XLP": "XLP", 
        "HYG": "HYG", "TLT": "TLT", "GLD": "GLD", "USO": "USO"
    }
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    try:
        data = yf.download(list(tickers_map.keys()), start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)
        
        if data.empty:
            return pd.DataFrame(columns=['Data'] + list(tickers_map.values()))

        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.get_level_values(0):
                df = data['Close'].copy()
            elif 'Close' in data.columns.get_level_values(1):
                df = data.xs('Close', level=1, axis=1)
            else:
                return pd.DataFrame(columns=['Data'] + list(tickers_map.values()))
        else:
            if 'Close' in data.columns:
                df = pd.DataFrame(data['Close'])
            else:
                df = data.copy()

        df = df.rename(columns=tickers_map).reset_index()
        col_date = [c for c in df.columns if str(c).lower() == 'date']
        if col_date:
            df = df.rename(columns={col_date[0]: 'Data'})
            df['Data'] = pd.to_datetime(df['Data']).dt.tz_localize(None).dt.normalize()
            
        cols_to_keep = ['Data'] + [c for c in df.columns if c in tickers_map.values()]
        return df[cols_to_keep]
        
    except Exception as e:
        print(f"Errore Critico YF Fetch: {e}")
        return pd.DataFrame(columns=['Data'] + list(tickers_map.values()))

def fetch_bridge_data():
    try:
        response = requests.get(GOOGLE_BRIDGE_URL, timeout=10)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        df.columns = df.columns.str.strip()
        
        col_mapping = {'Date': 'Data', 'Net_Liquidity': 'Net_Liquidity', 'M2': 'M2'}
        df = df.rename(columns=lambda x: col_mapping.get(x, x))
        
        if 'Data' not in df.columns:
            return pd.DataFrame(columns=["Data", "Net_Liquidity", "M2", "MOVE"])

        if pd.api.types.is_numeric_dtype(df['Data']):
            df['Data'] = pd.to_datetime(df['Data'], unit='D', origin='1899-12-30')
        else:
            df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
            
        df['Data'] = df['Data'].dt.normalize()
        
        for col in ['Net_Liquidity', 'M2', 'MOVE']:
            if col in df.columns: 
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        return df.dropna(subset=['Data'])
    except Exception:
        return pd.DataFrame(columns=["Data", "Net_Liquidity", "M2", "MOVE"])

def fetch_squeezemetrics_data():
    url = "https://squeezemetrics.com/monitor/static/DIX.csv"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize()
        df = df.dropna(subset=['date'])
        df = df.rename(columns={'date': 'Data', 'dix': 'DIX', 'gex': 'GEX'})
        df['DIX'] = df['DIX'] * 100
        return df[['Data', 'DIX', 'GEX']].sort_values('Data')
    except Exception:
        return pd.DataFrame(columns=['Data', 'DIX', 'GEX'])

def fetch_cboe_pc_ratio():
    url = "https://cdn.cboe.com/data/us/options/market_statistics/historical_data/totalpc.csv"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text), skiprows=2)
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce').dt.normalize()
        df = df.dropna(subset=['DATE'])
        df = df.rename(columns={'DATE': 'Data', 'P/C Ratio': 'P_C'})
        return df[['Data', 'P_C']].sort_values('Data')
    except Exception:
        return pd.DataFrame(columns=['Data', 'P_C'])

def calculate_rolling_zscore(series, window=252):
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std = series.rolling(window=window, min_periods=1).std(ddof=0)
    return np.where(rolling_std == 0, 0, (series - rolling_mean) / rolling_std)


# ==========================================================
# MODULO A: IDENTIFICATORE DI REGIME (I 9 PORTAFOGLI) & SOFTMAX
# ==========================================================

def fetch_regime_baskets_data(period="2y"):
    try:
        unique_tickers = sorted(list({ticker for basket in REGIME_BASKETS.values() for ticker in basket}))
        data = yf.download(tickers=unique_tickers, period=period, interval="1d", auto_adjust=True, progress=False)
        
        if data.empty:
            return pd.DataFrame()
            
        if isinstance(data.columns, pd.MultiIndex):
            if "Close" in data.columns.levels[0]:
                df = data["Close"].copy()
            else:
                df = data.xs(data.columns.levels[0][0], axis=1, level=0).copy()
        else:
            df = data.copy()
            
        return df.dropna(how="all").sort_index()
    except Exception as e:
        print(f"Errore Modulo A Data Fetching: {e}")
        return pd.DataFrame()

def calculate_regime_matrix(df_prices):
    if df_prices.empty or len(df_prices) < 63:
        return pd.DataFrame(), "Dati Insufficienti", 0.0

    matrix = []
    
    for regime, tickers in REGIME_BASKETS.items():
        valid_tickers = [t for t in tickers if t in df_prices.columns]
        if not valid_tickers:
            continue
            
        basket_prices = df_prices[valid_tickers].ffill()
        row_data = {"Regime": regime}
        
        for tf_label, days in TIMEFRAMES.items():
            if len(basket_prices) > days:
                p_now = basket_prices.iloc[-1]
                p_past = basket_prices.iloc[-(days + 1)]
                roc = ((p_now - p_past) / p_past) * 100.0
                row_data[tf_label] = float(np.nanmean(roc))
            else:
                row_data[tf_label] = np.nan
                
        matrix.append(row_data)

    df_matrix = pd.DataFrame(matrix).set_index("Regime")
    confidence_pct = 0.0
    
    if "Δ 1W" in df_matrix.columns and "Δ 1M" in df_matrix.columns:
        momentum_score = (df_matrix["Δ 1W"] + df_matrix["Δ 1M"]) / 2.0
        dominant = momentum_score.idxmax() if not momentum_score.dropna().empty else "N/D"
        
        # Trasformazione Softmax in probabilità (0-100)
        if dominant != "N/D":
            m_valid = momentum_score.dropna()
            if len(m_valid) > 1 and m_valid.std() > 0:
                z_scores = (m_valid - m_valid.mean()) / m_valid.std()
                exp_z = np.exp(z_scores)
                probs = (exp_z / exp_z.sum()) * 100.0
                confidence_pct = round(probs[dominant], 1)
            else:
                confidence_pct = 100.0
    else:
        dominant = "N/D"
        
    return df_matrix.round(2), dominant, confidence_pct


# ==========================================================
# MODULO B: POSIZIONAMENTO NEL CICLO ECONOMICO (LE 4 FASI)
# ==========================================================

def fetch_macro_cycle_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 4) 
    
    df_macro = pd.DataFrame()
    
    try:
        fred_series = {
            'DGS10': '10Y_Yield',
            'DGS2': '2Y_Yield',
            'DGS30': '30Y_Yield',
            'CPIAUCSL': 'CPI_Index'
        }
        df_fred = web.DataReader(list(fred_series.keys()), 'fred', start_date, end_date)
        df_fred = df_fred.rename(columns=fred_series)
        
        df_fred['CPI_Index'] = df_fred['CPI_Index'].ffill()
        df_fred['CPI_YoY'] = df_fred['CPI_Index'].pct_change(periods=252) * 100
        
        yf_data = yf.download(["HG=F", "GC=F"], start=start_date, end=end_date, progress=False)
        if isinstance(yf_data.columns, pd.MultiIndex):
            df_yf = yf_data['Close'].rename(columns={'HG=F': 'Copper', 'GC=F': 'Gold'})
        else:
            df_yf = yf_data.rename(columns={'HG=F': 'Copper', 'GC=F': 'Gold'})
            
        df_macro = pd.merge(df_fred, df_yf, left_index=True, right_index=True, how='inner')
        df_macro = df_macro.dropna(subset=['10Y_Yield', '2Y_Yield', 'Copper', 'Gold']).sort_index()
        return df_macro

    except Exception as e:
        print(f"Errore Modulo B Data Fetching: {e}")
        return pd.DataFrame()

def calculate_macro_cycle_phase(df_macro):
    if df_macro.empty or len(df_macro) < 252:
        return "DATI INSUFFICIENTI", {}
        
    df = df_macro.copy()
    
    df['Spread_10Y_2Y'] = df['10Y_Yield'] - df['2Y_Yield']
    df['Copper_Gold_Ratio'] = df['Copper'] / df['Gold']
    df['Real_Rates'] = df['10Y_Yield'] - df['CPI_YoY'].fillna(0)
    
    current_spread = df['Spread_10Y_2Y'].iloc[-1]
    is_inverted = current_spread < 0
    
    df['C_G_SMA200'] = df['Copper_Gold_Ratio'].rolling(window=200).mean()
    is_copper_gold_bullish = df['Copper_Gold_Ratio'].iloc[-1] > df['C_G_SMA200'].iloc[-1]
    
    df['Real_Rates_Z252'] = calculate_rolling_zscore(df['Real_Rates'], window=252)
    is_real_rates_breakout = df['Real_Rates_Z252'].iloc[-1] > 1.5 
    
    df['30Y_Z756'] = calculate_rolling_zscore(df['30Y_Yield'], window=756)
    is_debasement_risk = df['30Y_Z756'].iloc[-1] > 1.5
    
    if is_inverted and not is_copper_gold_bullish:
        fase = "CONTRAZIONE"
    elif is_inverted or is_real_rates_breakout:
        fase = "PICCO / STAGFLAZIONE"
    elif not is_inverted and is_copper_gold_bullish and not is_debasement_risk:
        fase = "ESPANSIONE"
    else:
        fase = "RIPRESA"
        
    metrics = {
        "Spread_10Y_2Y": round(current_spread, 2),
        "Copper_Gold_Trend": "Rialzista" if is_copper_gold_bullish else "Ribassista",
        "Real_Rates_Z": round(df['Real_Rates_Z252'].iloc[-1], 2),
        "30Y_Yield_Z": round(df['30Y_Z756'].iloc[-1], 2),
        "Debasement_Risk": is_debasement_risk
    }
    
    return fase, metrics

# ==========================================================
# MODULO: CALCOLO STATISTICO PROPENSIONE AL RISCHIO (RISK ON/OFF)
# ==========================================================

def calculate_risk_propensity(df_master):
    if df_master.empty or len(df_master) < 252:
        return None, "DATI INSUFFICIENTI"

    df_calc = df_master.tail(252).copy()
    
    required = ['XLY', 'XLP', 'SPY', 'RSP', 'HYG', 'TLT']
    if not all(col in df_calc.columns for col in required):
        return None, "DATI INSUFFICIENTI"

    df_calc['Risk_XLY_XLP'] = np.where(df_calc['XLP'] > 0, df_calc['XLY'] / df_calc['XLP'], np.nan)
    df_calc['Risk_SPY_RSP'] = np.where(df_calc['RSP'] > 0, df_calc['SPY'] / df_calc['RSP'], np.nan)
    df_calc['Risk_HYG_TLT'] = np.where(df_calc['TLT'] > 0, df_calc['HYG'] / df_calc['TLT'], np.nan)

    df_calc = df_calc.dropna(subset=['Risk_XLY_XLP', 'Risk_SPY_RSP', 'Risk_HYG_TLT'])
    if len(df_calc) < 200: 
        return None, "DATI STORICI CARENTI"

    z_xly = (df_calc['Risk_XLY_XLP'].iloc[-1] - df_calc['Risk_XLY_XLP'].mean()) / df_calc['Risk_XLY_XLP'].std(ddof=0)
    z_spy = (df_calc['Risk_SPY_RSP'].iloc[-1] - df_calc['Risk_SPY_RSP'].mean()) / df_calc['Risk_SPY_RSP'].std(ddof=0)
    z_hyg = (df_calc['Risk_HYG_TLT'].iloc[-1] - df_calc['Risk_HYG_TLT'].mean()) / df_calc['Risk_HYG_TLT'].std(ddof=0)

    avg_z = (z_xly + z_spy + z_hyg) / 3.0

    risk_on_prob = 0.5 * (1 + math.erf(avg_z / math.sqrt(2)))
    risk_on_pct = round(risk_on_prob * 100.0, 1)
    risk_off_pct = round(100.0 - risk_on_pct, 1)

    if risk_on_pct > 60:
        status = "RISK ON"
    elif risk_on_pct < 40:
        status = "RISK OFF"
    else:
        status = "NEUTRAL"

    metrics = {
        "Risk_On_Pct": risk_on_pct,
        "Risk_Off_Pct": risk_off_pct,
        "Status": status,
        "Z_Avg": round(avg_z, 2)
    }

    return metrics, None
