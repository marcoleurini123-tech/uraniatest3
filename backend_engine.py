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

COLUMNS = [
    "Data", "VIX1D", "VIX9D", "VIX", "VIX3M", "VIX6M", "VIX1Y", "VVIX", "MOVE", "SKEW", 
    "DXY", "DIX", "GEX", "SPY", "RSP", "HYG", "XLY", "XLP", "TLT", "P_C", "GLD", "USO", 
    "Net_Liquidity", "M2"
]

# MATRICE RIGOROSA: 10 Portafogli (Inclusi Modelli Debasement)
REGIME_BASKETS = {
    "GOLDILOCKS ECONOMY": ["QQQ", "XLK", "XLY", "IEF", "SMH"],
    "RECESSION": ["TLT", "SHY", "XLU", "XLP", "GLD"],
    "STAGFLATION": ["GLD", "DBC", "XLE", "TIP", "XLU"],
    "REFLATION": ["XLI", "XLF", "IWM", "EEM", "DBC"],
    "DISINFLATION/SOFT LANDING": ["TLT", "LQD", "QQQ", "VTI", "GLD"],
    "DOLLAR WEAKNESS/GLOBAL REBALANCING": ["EEM", "FXF", "GLD", "IXUS", "DBC"],
    "DEFLATION": ["TLT", "BIL", "SHY", "XLP", "XLU"],
    "DOLLAR WEAKNESS/GLOBAL REBALANCING +BITCOIN": ["EEM", "FXF", "GLD", "IXUS", "IBIT"],
    "DEBASEMENT AGGRESSIVO": ["GLD", "XME", "COPX", "EEM", "IBIT"],
    "DEBASEMENT (SENZA BITCOIN)": ["GLD", "XME", "COPX", "EEM", "VDST.MI"]
}

# OFFSET RIGIDI: Misurazione in timedelta per riproduzione algoritmi proprietari (Giorni lineari)
TIMEFRAMES_CALENDAR = {
    "Δ 1D": "session", 
    "Δ 1W": timedelta(days=7),
    "Δ 1M": timedelta(days=30),
    "Δ 3M": timedelta(days=90),
    "Δ 6M": timedelta(days=180),
    "Δ 1Y": timedelta(days=365),
    "Δ 2Y": timedelta(days=730),
    "Δ 3Y": timedelta(days=1095),
    "Δ 5Y": timedelta(days=1825)
}

# ==========================================================
# FASE 1: FETCHING DATI EOD - RIGORE ASSOLUTO
# ==========================================================
def load_db():
    if os.path.exists(DB_FILE):
        df = pd.read_csv(DB_FILE)
        df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
        if df['Data'].dt.tz is not None:
            df['Data'] = df['Data'].dt.tz_localize(None)
        df['Data'] = df['Data'].dt.normalize()
        for col in COLUMNS:
            if col not in df.columns: df[col] = np.nan
        num_cols = [c for c in COLUMNS if c != "Data"]
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return df.dropna(subset=['Data']).sort_values("Data")
    return pd.DataFrame(columns=COLUMNS)

def save_db(df):
    if df.empty: return
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
        if data.empty: return pd.DataFrame(columns=['Data'] + list(tickers_map.values()))
        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.get_level_values(0): df = data['Close'].copy()
            elif 'Close' in data.columns.get_level_values(1): df = data.xs('Close', level=1, axis=1)
            else: return pd.DataFrame(columns=['Data'] + list(tickers_map.values()))
        else:
            if 'Close' in data.columns: df = pd.DataFrame(data['Close'])
            else: df = data.copy()
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
        if 'Data' not in df.columns: return pd.DataFrame(columns=["Data", "Net_Liquidity", "M2", "MOVE"])
        if pd.api.types.is_numeric_dtype(df['Data']): df['Data'] = pd.to_datetime(df['Data'], unit='D', origin='1899-12-30')
        else: df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
        df['Data'] = df['Data'].dt.normalize()
        for col in ['Net_Liquidity', 'M2', 'MOVE']:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
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
# FASE 2: MATRICE REGIMI (ALGORITMO GOOGLE FINANCE)
# ==========================================================
def fetch_regime_baskets_data(period="10y"):
    try:
        unique_tickers = sorted(list({ticker for basket in REGIME_BASKETS.values() for ticker in basket}))
        data = yf.download(tickers=unique_tickers, period=period, interval="1d", auto_adjust=False, progress=False)
        if data.empty: return pd.DataFrame()
        
        if isinstance(data.columns, pd.MultiIndex):
            if "Close" in data.columns.levels[0]: df = data["Close"].copy()
            else: df = data.xs("Close", axis=1, level=0).copy()
        else: 
            if 'Close' in data.columns:
                df = pd.DataFrame(data['Close'])
                df.columns = [unique_tickers[0]]
            else:
                df = data.copy()
        
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index = df.index.normalize()
        
        return df.dropna(how="all").sort_index()
    except Exception:
        return pd.DataFrame()

def calculate_regime_matrix(df_prices):
    if df_prices.empty or len(df_prices) < 5:
        return pd.DataFrame(), "Dati Insufficienti", 0.0

    matrix = []
    today = pd.Timestamp(datetime.now().date())
    
    for regime, tickers in REGIME_BASKETS.items():
        valid_tickers = [t for t in tickers if t in df_prices.columns]
        if not valid_tickers: continue
        
        basket_prices = df_prices[valid_tickers].ffill()
        
        valid_current_slice = basket_prices.loc[:today]
        if valid_current_slice.empty: continue
        p_now = valid_current_slice.iloc[-1]
        
        row_data = {"Regime": regime}
        
        for tf_label, offset in TIMEFRAMES_CALENDAR.items():
            if tf_label == "Δ 1D":
                if len(valid_current_slice) >= 2:
                    p_past = valid_current_slice.iloc[-2]
                else:
                    p_past = pd.Series(np.nan, index=basket_prices.columns)
            else:
                target_date = today - offset
                p_past_dict = {}
                
                for t in basket_prices.columns:
                    series = basket_prices[t].dropna()
                    if series.empty:
                        p_past_dict[t] = np.nan
                        continue
                    
                    first_idx = series.index[0]
                    if first_idx > target_date:
                        p_past_dict[t] = np.nan
                    else:
                        slice_forward = series.loc[target_date:]
                        if not slice_forward.empty:
                            first_valid_date = slice_forward.index[0]
                            if (first_valid_date - pd.Timestamp(target_date)).days <= 7:
                                p_past_dict[t] = slice_forward.iloc[0]
                            else:
                                p_past_dict[t] = np.nan
                        else:
                            p_past_dict[t] = np.nan
                            
                p_past = pd.Series(p_past_dict)
            
            roc = ((p_now - p_past) / p_past) * 100.0
            valid_roc = roc.dropna()
            
            if not valid_roc.empty:
                row_data[tf_label] = float(valid_roc.mean())
            else:
                row_data[tf_label] = np.nan
                
        matrix.append(row_data)

    if not matrix:
        return pd.DataFrame(), "Dati Insufficienti", 0.0

    df_matrix = pd.DataFrame(matrix).set_index("Regime")
    confidence_pct = 0.0
    dominant = "N/D"

    # MATEMATICA STRUTTURALE: Sradicamento del rumore a breve termine.
    # Il regime dominante si misura escludendo 1W e 1M, ponderando oggettivamente 3M (60%) e 6M (40%)
    if "Δ 3M" in df_matrix.columns and "Δ 6M" in df_matrix.columns:
        momentum_score = (df_matrix["Δ 3M"] * 0.6) + (df_matrix["Δ 6M"] * 0.4)
        m_valid = momentum_score.dropna()
        if not m_valid.empty:
            dominant = m_valid.idxmax()
            if len(m_valid) > 1 and m_valid.std() > 0:
                z_scores = (m_valid - m_valid.mean()) / m_valid.std()
                exp_z = np.exp(z_scores)
                probs = (exp_z / exp_z.sum()) * 100.0
                confidence_pct = round(probs[dominant], 1)
            else:
                confidence_pct = 100.0
                
    return df_matrix.round(2), dominant, confidence_pct

# ==========================================================
# FASE 3: MOTORE CICLO ECONOMICO E VETO
# ==========================================================
def fetch_macro_cycle_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 4) 
    try:
        fred_series = {'DGS10': '10Y_Yield', 'DGS2': '2Y_Yield', 'DGS30': '30Y_Yield'}
        df_fred = web.DataReader(list(fred_series.keys()), 'fred', start_date, end_date)
        df_fred = df_fred.rename(columns=fred_series).ffill()
        
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
    if df_macro.empty or len(df_macro) < 252:
        return "DATI INSUFFICIENTI", "N/D", False, {}
        
    df = df_macro.copy()
    df['Spread_10Y_2Y'] = df['10Y_Yield'] - df['2Y_Yield']
    current_spread = df['Spread_10Y_2Y'].iloc[-1]
    
    df['Copper_Gold_Ratio'] = df['Copper'] / df['Gold']
    window = 40
    if len(df) >= window:
        y_vals = df['Copper_Gold_Ratio'].iloc[-window:].values
        x_vals = np.arange(len(y_vals))
        slope, _, _, _, _ = linregress(x_vals, y_vals)
        cg_slope = slope / df['Copper_Gold_Ratio'].iloc[-window] * 1000
    else: cg_slope = 0.0

    if 'TIPS_ETF' in df.columns:
        mean_tips = df['TIPS_ETF'].rolling(252).mean().iloc[-1]
        std_tips = df['TIPS_ETF'].rolling(252).std().iloc[-1]
        z_real_rates = - (df['TIPS_ETF'].iloc[-1] - mean_tips) / (std_tips + 1e-9)
    else: z_real_rates = 0.0

    mean_30y = df['30Y_Yield'].rolling(252).mean().iloc[-1]
    std_30y = df['30Y_Yield'].rolling(252).std().iloc[-1]
    z_30y = (df['30Y_Yield'].iloc[-1] - mean_30y) / (std_30y + 1e-9)

    if current_spread > 0 and cg_slope > 0: raw_phase = "Espansione"
    elif current_spread > 0 and cg_slope <= 0: raw_phase = "Ripresa"
    elif current_spread <= 0 and cg_slope > 0: raw_phase = "Picco / Stagflazione"
    else: raw_phase = "Contrazione"

    veto_applied = False
    final_phase = raw_phase
    regimi_antitetici = ["DEBASEMENT (SENZA BITCOIN)", "DEBASEMENT AGGRESSIVO", "STAGFLATION"]
    
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
# FASE 4: Z-SCORE PROPENSIONE AL RISCHIO E HARD OVERRIDE
# ==========================================================
def calculate_risk_propensity(df_master):
    if df_master.empty or len(df_master) < 252:
        return None, "DATI INSUFFICIENTI"
    df_calc = df_master.tail(252).copy()
    required = ['XLY', 'XLP', 'SPY', 'RSP', 'HYG', 'TLT']
    if not all(col in df_calc.columns for col in required): return None, "DATI INSUFFICIENTI"
    
    df_calc['Risk_XLY_XLP'] = np.where(df_calc['XLP'] > 0, df_calc['XLY'] / df_calc['XLP'], np.nan)
    df_calc['Risk_SPY_RSP'] = np.where(df_calc['RSP'] > 0, df_calc['SPY'] / df_calc['RSP'], np.nan)
    df_calc['Risk_HYG_TLT'] = np.where(df_calc['TLT'] > 0, df_calc['HYG'] / df_calc['TLT'], np.nan)
    
    df_calc = df_calc.dropna(subset=['Risk_XLY_XLP', 'Risk_SPY_RSP', 'Risk_HYG_TLT'])
    if len(df_calc) < 200: return None, "DATI STORICI CARENTI"

    # Sicurezza Matematica: +1e-9 previene DivisionByZero 
    z_xly = (df_calc['Risk_XLY_XLP'].iloc[-1] - df_calc['Risk_XLY_XLP'].mean()) / (df_calc['Risk_XLY_XLP'].std(ddof=0) + 1e-9)
    z_spy = (df_calc['Risk_SPY_RSP'].iloc[-1] - df_calc['Risk_SPY_RSP'].mean()) / (df_calc['Risk_SPY_RSP'].std(ddof=0) + 1e-9)
    z_hyg = (df_calc['Risk_HYG_TLT'].iloc[-1] - df_calc['Risk_HYG_TLT'].mean()) / (df_calc['Risk_HYG_TLT'].std(ddof=0) + 1e-9)
    
    # Integrazione Modello Struttura VIX (Backwardation = Risk Off)
    z_vix = 0.0
    if 'VIX' in df_calc.columns and 'VIX3M' in df_calc.columns:
        df_calc['VIX_Struct'] = np.where(df_calc['VIX3M'] > 0, df_calc['VIX'] / df_calc['VIX3M'], np.nan)
        vix_s = df_calc['VIX_Struct'].dropna()
        if len(vix_s) > 100:
            z_vix = -((vix_s.iloc[-1] - vix_s.mean()) / (vix_s.std(ddof=0) + 1e-9))
            
    # Integrazione Modello Stress Obbligazionario (Elevato MOVE = Risk Off)
    z_move = 0.0
    if 'MOVE' in df_calc.columns:
        move_s = df_calc['MOVE'].dropna()
        if len(move_s) > 100:
             z_move = -((move_s.iloc[-1] - move_s.mean()) / (move_s.std(ddof=0) + 1e-9))

    # Ponderazione aggregata a 5 Fattori
    avg_z = (z_xly + z_spy + z_hyg + z_vix + z_move) / 5.0

    risk_on_prob = 0.5 * (1 + math.erf(avg_z / math.sqrt(2)))
    risk_on_pct = round(risk_on_prob * 100.0, 1)
    risk_off_pct = round(100.0 - risk_on_pct, 1)

    if risk_on_pct > 60: status = "RISK ON"
    elif risk_on_pct < 40: status = "RISK OFF"
    else: status = "NEUTRAL"

    metrics = {
        "Risk_On_Pct": risk_on_pct,
        "Risk_Off_Pct": risk_off_pct,
        "Status": status,
        "Z_Avg": round(avg_z, 2)
    }
    return metrics, None

def evaluate_risk_override(df_db):
    if df_db.empty: return False, ["Dati Mancanti"], np.nan, np.nan
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
                if momentum_1m < -0.065: net_liq_contraction = True

    trigger_skew = (not np.isnan(skew_val)) and (skew_val >= 140.0)
    trigger_vix = (not np.isnan(vix_val)) and (vix_val >= 30.0)
    is_risk_off = trigger_skew or trigger_vix or net_liq_contraction
    
    reasons = []
    if trigger_skew: reasons.append(f"SKEW Critico: {skew_val:.1f} (> 140.0)")
    if trigger_vix: reasons.append(f"VIX Panico: {vix_val:.1f} (> 30.0)")
    if net_liq_contraction: reasons.append("Contrazione mensile netta della Liquidità M2 > 6.5%")
    
    return is_risk_off, reasons, skew_val, vix_val
