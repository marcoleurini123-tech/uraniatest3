import pandas as pd
import numpy as np
import requests
import zipfile
import io
from datetime import datetime

# ==========================================================
# DIZIONARIO ASSET CFTC (Derivato da Page 4)
# Mappatura inversa per decodificare il database grezzo
# ==========================================================
CFTC_MAPPING = {
    # Equities
    "E-MINI S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE": "SPX",
    "E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE": "SPX",
    "VIX FUTURES - CBOE FUTURES EXCHANGE": "VIX",
    "NASDAQ-100 STOCK INDEX (MINI) - CHICAGO MERCANTILE EXCHANGE": "Nasdaq",
    "NASDAQ MINI - CHICAGO MERCANTILE EXCHANGE": "Nasdaq",
    "RUSSELL 2000 MINI INDEX FUTURE - ICE FUTURES U.S.": "Russell 2000",
    "RUSSELL E-MINI - CHICAGO MERCANTILE EXCHANGE": "Russell 2000",
    "NIKKEI STOCK AVERAGE - CHICAGO MERCANTILE EXCHANGE": "Nikkei",
    
    # Rates
    "10-YEAR U.S. TREASURY NOTES - CHICAGO BOARD OF TRADE": "10Y UST",
    "UST 10Y NOTE - CHICAGO BOARD OF TRADE": "10Y UST",
    "2-YEAR U.S. TREASURY NOTES - CHICAGO BOARD OF TRADE": "2Y UST",
    "2Y NOTE - CHICAGO BOARD OF TRADE": "2Y UST",
    "5-YEAR U.S. TREASURY NOTES - CHICAGO BOARD OF TRADE": "5Y UST",
    "U.S. TREASURY BONDS - CHICAGO BOARD OF TRADE": "UST Bonds",
    
    # Currencies
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
    
    # Commodities
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

# ==========================================================
# FASE 1: ESTRAZIONE DATI UFFICIALI CFTC (Regola 1)
# ==========================================================
def fetch_cftc_data(years_back=4):
    """
    Estrae i file ZIP 'Legacy Futures Only' dai server della CFTC.
    Scarica 4 anni di dati per permettere un calcolo Z-Score solido a 156 settimane (3 anni).
    Nessun dato fittizio in caso di errore.
    """
    current_year = datetime.now().year
    years = [str(current_year - i) for i in range(years_back)]
    
    dfs = []
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    for year in years:
        url = f"https://www.cftc.gov/files/dea/history/dea_fut_txt_{year}.zip"
        try:
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    filename = z.namelist()[0]
                    with z.open(filename) as f:
                        df = pd.read_csv(f, low_memory=False)
                        dfs.append(df)
        except Exception as e:
            print(f"🚨 ERRORE RETE CFTC ({year}): {e}")
            
    if not dfs:
        raise ValueError("🚨 IMPOSSIBILE SCARICARE I DATI CFTC. Connessione interrotta.")
        
    df_raw = pd.concat(dfs, ignore_index=True)
    
    cols = {
        'Market and Exchange Names': 'Asset_Raw',
        'Report_Date_as_MM_DD_YYYY': 'Data',
        'NonComm_Positions_Long_All': 'NC_Long',
        'NonComm_Positions_Short_All': 'NC_Short',
        'Comm_Positions_Long_All': 'Comm_Long',
        'Comm_Positions_Short_All': 'Comm_Short'
    }
    
    # Filtro colonne esistenti
    valid_cols = {k: v for k, v in cols.items() if k in df_raw.columns}
    df_clean = df_raw[list(valid_cols.keys())].rename(columns=valid_cols)
    
    df_clean['Data'] = pd.to_datetime(df_clean['Data'])
    df_clean = df_clean.sort_values('Data').dropna(subset=['Data'])
    
    # Mappatura Asset: Tiene solo i contratti presenti nel dizionario
    df_clean['Asset'] = df_clean['Asset_Raw'].map(CFTC_MAPPING)
    df_clean = df_clean.dropna(subset=['Asset'])
    
    # Calcolo Posizionamento Netto Assoluto
    df_clean['Net_NC'] = df_clean['NC_Long'] - df_clean['NC_Short']
    df_clean['Net_Comm'] = df_clean['Comm_Long'] - df_clean['Comm_Short']
    
    return df_clean

# ==========================================================
# FASE 2: MOTORE Z-SCORE 156 SETTIMANE E STELLE (Regola 2)
# ==========================================================
def calculate_cot_zscores(df_cot, window_weeks=156):
    """
    Calcola lo Z-Score storico sulle Posizioni Nette di Commercials e Non-Commercials.
    Identifica gli estremi (> 1.8 o < -1.8) assegnando la ⭐.
    """
    if df_cot.empty:
        return pd.DataFrame()

    df = df_cot.sort_values(by=['Asset', 'Data']).copy()
    
    # Devono esserci almeno 52 settimane di storico per evitare rumore statistico
    min_obs = 52 
    
    # Calcolo Z-Score Non-Commercials
    df['NC_Mean'] = df.groupby('Asset')['Net_NC'].transform(lambda x: x.rolling(window_weeks, min_periods=min_obs).mean())
    df['NC_Std'] = df.groupby('Asset')['Net_NC'].transform(lambda x: x.rolling(window_weeks, min_periods=min_obs).std())
    df['Z_NC'] = (df['Net_NC'] - df['NC_Mean']) / (df['NC_Std'] + 1e-9)
    
    # Calcolo Z-Score Commercials
    df['Comm_Mean'] = df.groupby('Asset')['Net_Comm'].transform(lambda x: x.rolling(window_weeks, min_periods=min_obs).mean())
    df['Comm_Std'] = df.groupby('Asset')['Net_Comm'].transform(lambda x: x.rolling(window_weeks, min_periods=min_obs).std())
    df['Z_Comm'] = (df['Net_Comm'] - df['Comm_Mean']) / (df['Comm_Std'] + 1e-9)
    
    # Estraiamo l'ultima riga disponibile per ogni asset (L'ultimo report CFTC)
    latest = df.groupby('Asset').tail(1).copy()
    latest = latest.dropna(subset=['Z_NC', 'Z_Comm'])
    
    # Arrotondamento per UI
    latest['Z_NC'] = latest['Z_NC'].round(2)
    latest['Z_Comm'] = latest['Z_Comm'].round(2)
    
    # Assegnazione Stella Estremi (Soglia +/- 1.8)
    latest['Alert_NC'] = np.where(latest['Z_NC'].abs() >= 1.8, "⭐", "")
    latest['Alert_Comm'] = np.where(latest['Z_Comm'].abs() >= 1.8, "⭐", "")
    
    cols_to_return = ['Data', 'Asset', 'Net_NC', 'Z_NC', 'Alert_NC', 'Net_Comm', 'Z_Comm', 'Alert_Comm']
    
    # Ordiniamo dal più estremo al meno estremo sui Commercials
    latest = latest[cols_to_return].sort_values(by='Z_Comm', key=abs, ascending=False).reset_index(drop=True)
    
    return latest

# ==========================================================
# ESECUZIONE DI TEST (Da terminale locale)
# ==========================================================
if __name__ == "__main__":
    print("Inizio estrazione dai server CFTC (potrebbe richiedere 10-20 secondi)...")
    try:
        df_raw = fetch_cftc_data(years_back=4)
        print(f"Righe grezze estratte e mappate: {len(df_raw)}")
        
        df_zscore = calculate_cot_zscores(df_raw, window_weeks=156)
        print("\n=== ULTIMO REPORT COT (Z-SCORE 3 ANNI) ===")
        print(df_zscore.to_string())
    except Exception as e:
        print(e)
