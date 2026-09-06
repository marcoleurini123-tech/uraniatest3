import pandas as pd
import numpy as np
import requests
import io
import yfinance as yf
import os
import math
from datetime import datetime, timedelta
import pandas_datareader.data as web
from scipy.stats import linregress

DB_FILE = "macro_database.csv"
GOOGLE_BRIDGE_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSeeY57SBwd6BftA2Bq8C0nyzzT3wj9WRWOihDF7QE-COPXhC4r2RN_k_BRgZke1nU2BbKT8oRlsXOX/pub?gid=1412711569&single=true&output=csv"

# Definizione rigida delle matrici ammesse nel Database
COLUMNS = [
    "Data", "VIX1D", "VIX9D", "VIX", "VIX3M", "VIX6M", "VIX1Y", "VVIX", "MOVE", "SKEW", 
    "DXY", "DIX", "GEX", "SPY", "RSP", "HYG", "XLY", "XLP", "TLT", "P_C", "GLD", "USO", 
    "Net_Liquidity", "M2"
]

# --- COSTANTI MODULO A (REGIMI MACRO) - NOMENCLATURA PROPRIETARIA ---
REGIME_BASKETS = {
    "Crescita Equilibrata": ["SPY", "QQQ", "IWM", "LQD"],
    "Contrazione Ciclica": ["TLT", "IEF", "ZROZ"],
    "Stagflazione": ["GLD", "PDBC", "TIP"],
    "Reflazione": ["XLE", "XLF", "XLI", "COPX"],
    "Disinflazione / Soft Landing": ["IEF", "LQD", "XLP"],
    "Indebolimento Valutario / Global Rebalance": ["EEM", "VEA", "GLD"],
    "Deflazione": ["BIL", "SHY", "GOVT"],
    "Svalutazione Monetaria Base": ["GLD", "SLV", "GDX"],
    "Svalutazione Monetaria Aggressiva": ["IBIT", "MSTR", "GLD", "SLV"]
}

# Estensione orizzonte temporale a 5 Anni (1260 giorni lavorativi)
TIMEFRAMES = {
    "Δ 1D": 1, "Δ 1W": 5, "Δ 1M": 21, "Δ 3M": 63, 
    "Δ 6M": 126, "Δ 1Y": 252, "Δ 2Y": 504, "Δ 3Y": 756, "Δ 5Y": 1260
}

# ==========================================================
# FUNZIONI DI BASE, DATABASE E DATA FETCHING (NESSUNA ALLUCINAZIONE)
# ==========================================================
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
    if df.empty:
        return
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
        
    except Exception:
        return pd.DataFrame(columns=['Data'] + list(tickers_map.values()))

def fetch_bridge_data():
    try:
        response = requests.get(GOOGLE_BRIDGE_URL, timeout=10)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        df.columns = df.columns.str.strip()
        
        col_mapping = {'Date': 'Data', 'Net_Liquidity': 'Net_Liquidity', 'M2': 'M2', 'MOVE': 'MOVE'}
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
    """Calcolo rigoroso Z-Score storici."""
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std = series.rolling(window=window, min_periods=1).std(ddof=0)
    return np.where(rolling_std == 0, 0, (series - rolling_mean) / rolling_std)


# ==========================================================
# MODULO A: MATRICE DEI REGIMI & Z-SCORE PREDOMINANTE
# ==========================================================
def fetch_regime_baskets_data(period="5y"):
    """Recupera fino a 5 anni di storico per i 9 portafogli macro."""
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
    except Exception:
        return pd.DataFrame()

def calculate_regime_matrix(df_prices):
    """
    Calcola la matrice dei ritorni equipesati fino a 5Y.
    Determina il Regime Predominante calcolando lo Z-Score cross-sezionale sui rendimenti 1W e 1M.
    """
    if df_prices.empty:
        return pd.DataFrame(), "Dati Insufficienti", pd.Series()

    matrix = []
    
    for regime, tickers in REGIME_BASKETS.items():
        valid_tickers = [t for t in tickers if t in df_prices.columns]
        if not valid_tickers:
            continue
            
        basket_prices = df_prices[valid_tickers].ffill()
        daily_returns = basket_prices.pct_change()
        eq_daily_ret = daily_returns.mean(axis=1)
        cumulative_idx = (1 + eq_daily_ret).cumprod()
        
        row_data = {"Regime": regime}
        
        for tf_label, days in TIMEFRAMES.items():
            if len(cumulative_idx) > days:
                p_now = cumulative_idx.iloc[-1]
                p_past = cumulative_idx.iloc[-(days + 1)]
                roc = ((p_now - p_past) / p_past) * 100.0
                row_data[tf_label] = float(roc)
            else:
                row_data[tf_label] = np.nan
                
        matrix.append(row_data)

    if not matrix:
        return pd.DataFrame(), "Dati Insufficienti", pd.Series()

    df_matrix = pd.DataFrame(matrix).set_index("Regime")
    
    predominant_regime = "N/D"
    z_combined = pd.Series(dtype=float)

    if "Δ 1W" in df_matrix.columns and "Δ 1M" in df_matrix.columns:
        perf_1w = df_matrix["Δ 1W"].astype(float)
        perf_1m = df_matrix["Δ 1M"].astype(float)
        
        if not perf_1w.isna().all() and not perf_1m.isna().all():
            z_1w = (perf_1w - perf_1w.mean()) / (perf_1w.std() + 1e-9)
            z_1m = (perf_1m - perf_1m.mean()) / (perf_1m.std() + 1e-9)
            z_combined = (z_1w + z_1m) / 2.0
            predominant_regime = z_combined.idxmax()
            
    return df_matrix.round(2), predominant_regime, z_combined


# ==========================================================
# MODULO B: CICLO ECONOMICO & REGOLA DEL VETO INCROCIATO
# ==========================================================
def fetch_macro_cycle_data():
    """Recupera dati di ciclo (Spread Curva, Rame, Oro, TIPS, 30Y Treasury)"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 4) 
    
    try:
        fred_series = {
            'DGS10': '10Y_Yield',
            'DGS2': '2Y_Yield',
            'DGS30': '30Y_Yield'
        }
        df_fred = web.DataReader(list(fred_series.keys()), 'fred', start_date, end_date)
        df_fred = df_fred.rename(columns=fred_series).ffill()
        
        # Sostituiamo il CPI con il proxy TIPS per ottenere i tassi reali in tempo reale tramite mercato
        yf_data = yf.download(["HG=F", "GC=F", "TIP"], start=start_date, end=end_date, progress=False)
        if isinstance(yf_data.columns, pd.MultiIndex):
            df_yf = yf_data['Close'].rename(columns={'HG=F': 'Copper', 'GC=F': 'Gold', 'TIP': 'TIPS_ETF'})
        else:
            df_yf = yf_data.rename(columns={'HG=F': 'Copper', 'GC=F': 'Gold', 'TIP': 'TIPS_ETF'})
            
        df_macro = pd.merge(df_fred, df_yf, left_index=True, right_index=True, how='inner')
        return df_macro.dropna(subset=['10Y_Yield', '2Y_Yield', 'Copper', 'Gold']).sort_index()
    except Exception:
        return pd.DataFrame()

def calculate_macro_cycle_phase(df_macro, predominant_regime):
    """
    Determina la fase (Ripresa, Espansione, Picco/Stagflazione, Contrazione).
    Input esclusivi matematici + Applicazione Hard Override del Veto.
    """
    if df_macro.empty or len(df_macro) < 252:
        return "DATI INSUFFICIENTI", "N/D", False, {}
        
    df = df_macro.copy()
    
    # 1. Spread 10Y - 2Y
    df['Spread_10Y_2Y'] = df['10Y_Yield'] - df['2Y_Yield']
    current_spread = df['Spread_10Y_2Y'].iloc[-1]
    
    # 2. Pendenza Copper / Gold (Ultimi 40 GG lavorativi)
    df['Copper_Gold_Ratio'] = df['Copper'] / df['Gold']
    window = 40
    if len(df) >= window:
        y_vals = df['Copper_Gold_Ratio'].iloc[-window:].values
        x_vals = np.arange(len(y_vals))
        slope, _, _, _, _ = linregress(x_vals, y_vals)
        cg_slope = slope / df['Copper_Gold_Ratio'].iloc[-window] * 1000
    else:
        cg_slope = 0.0

    # 3. Z-Score Tassi Reali (Proxy ETF TIP, direzione inversa)
    if 'TIPS_ETF' in df.columns:
        mean_tips = df['TIPS_ETF'].rolling(252).mean().iloc[-1]
        std_tips = df['TIPS_ETF'].rolling(252).std().iloc[-1]
        z_real_rates = - (df['TIPS_ETF'].iloc[-1] - mean_tips) / (std_tips + 1e-9)
    else:
        z_real_rates = 0.0

    # 4. Z-Score 30Y Treasury
    mean_30y = df['30Y_Yield'].rolling(252).mean().iloc[-1]
    std_30y = df['30Y_Yield'].rolling(252).std().iloc[-1]
    z_30y = (df['30Y_Yield'].iloc[-1] - mean_30y) / (std_30y + 1e-9)

    # Logica di base
    if current_spread > 0 and cg_slope > 0:
        raw_phase = "Espansione"
    elif current_spread > 0 and cg_slope <= 0:
        raw_phase = "Ripresa"
    elif current_spread <= 0 and cg_slope > 0:
        raw_phase = "Picco / Stagflazione"
    else:
        raw_phase = "Contrazione"

    # ==========================================================================
    # REGOLA DEL VETO INCROCIATO RIGIDO
    # Se il Regime Predominante è Svalutazione o Stagflazione, "Ripresa" è interdetta.
    # ==========================================================================
    veto_applied = False
    final_phase = raw_phase
    regimi_antitetici = ["Svalutazione Monetaria Base", "Svalutazione Monetaria Aggressiva", "Stagflazione"]
    
    if predominant_regime in regimi_antitetici and raw_phase == "Ripresa":
        veto_applied = True
        final_phase = "Picco / Stagflazione" if cg_slope >= 0 else "Contrazione"

    metrics = {
        "Spread_10Y_2Y": round(current_spread, 2),
        "Pendenza_Cu_Au_40D": round(cg_slope, 3),
        "Z_Score_Tassi_Reali": round(z_real_rates, 2),
        "Z_Score_30Y_Yield": round(z_30y, 2)
    }
    
    return final_phase, raw_phase, veto_applied, metrics


# ==========================================================
# MODULO C: PROTOCOLLO DI HARD OVERRIDE (RISK MANAGEMENT)
# ==========================================================
def evaluate_risk_override(df_db):
    """
    Monitora SKEW, VIX e contrazione della liquidità.
    Restituisce Boolean per Risk Off, array di cause ed i valori attuali.
    """
    if df_db.empty:
        return False, ["Dati Mancanti"], np.nan, np.nan
        
    df_clean = df_db.sort_values("Data").ffill()
    
    skew_val = df_clean['SKEW'].iloc[-1] if 'SKEW' in df_clean.columns else np.nan
    vix_val = df_clean['VIX'].iloc[-1] if 'VIX' in df_clean.columns else np.nan
    
    net_liq_contraction = False
    if 'Net_Liquidity' in df_clean.columns:
        valid_liq = df_clean['Net_Liquidity'].dropna()
        if len(valid_liq) >= 21:
            liq_now = valid_liq.iloc[-1]
            liq_past = valid_liq.iloc[-21]
            if liq_past > 0:
                momentum_1m = (liq_now - liq_past) / liq_past
                if momentum_1m < -0.065: # Contrazione superiore al 6.5% mese su mese
                    net_liq_contraction = True

    trigger_skew = (not np.isnan(skew_val)) and (skew_val >= 140.0)
    trigger_vix = (not np.isnan(vix_val)) and (vix_val >= 30.0)
    
    is_risk_off = trigger_skew or trigger_vix or net_liq_contraction
    
    reasons = []
    if trigger_skew:
        reasons.append(f"SKEW Critico: {skew_val:.1f} (> 140.0)")
    if trigger_vix:
        reasons.append(f"VIX Panico: {vix_val:.1f} (> 30.0)")
    if net_liq_contraction:
        reasons.append("Contrazione mensile netta della Liquidità M2 > 6.5%")
        
    return is_risk_off, reasons, skew_val, vix_val
