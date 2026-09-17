import pandas as pd
import zipfile
import os
import glob

# ==========================================================
# DIZIONARIO ASSET CFTC (Report Legacy)
# ==========================================================
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

def build_offline_database():
    """
    Regola 1 & Regola 4: Estrazione dati offline con parser dinamico.
    Rileva automaticamente le variazioni di naming della CFTC negli storici.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(base_dir, "cftc_raw")
    
    if not os.path.exists(target_dir):
        raise FileNotFoundError(f"🚨 DIRECTORY MANCANTE: La cartella '{target_dir}' non esiste.")
        
    zip_files = glob.glob(os.path.join(target_dir, "deacot*.zip")) + glob.glob(os.path.join(target_dir, "dea_fut_txt_*.zip"))
    
    if not zip_files:
        raise FileNotFoundError(f"🚨 DATI MANCANTI: Nessun archivio rilevato in '{target_dir}'.")
        
    dfs = []
    print("Inizio decodifica tensori offline (Legacy Reports)...")
    
    for z_path in sorted(zip_files):
        print(f"-> Lettura vettore: {os.path.basename(z_path)}")
        try:
            with zipfile.ZipFile(z_path, 'r') as z:
                filename = z.namelist()[0]
                with z.open(filename) as f:
                    df = pd.read_csv(f, low_memory=False)
                    # Normalizzazione rigorosa: tutto minuscolo, niente spazi vuoti
                    df.columns = df.columns.str.strip().str.lower()
                    dfs.append(df)
        except Exception as e:
            print(f"🚨 ERRORE I/O sul file {z_path}: {e}")
            
    if not dfs:
        raise ValueError("🚨 IMPOSSIBILE COSTRUIRE LA MATRICE DATI.")
        
    df_raw = pd.concat(dfs, ignore_index=True)
    
    # ---------------------------------------------------------
    # PARSER DINAMICO DELLE COLONNE (Pattern Matching)
    # ---------------------------------------------------------
    col_map = {}
    
    try:
        # Asset Name
        asset_col = [c for c in df_raw.columns if 'market and exchange' in c or 'contract_market' in c][0]
        col_map[asset_col] = 'Asset_Raw'
        
        # Data
        date_col = [c for c in df_raw.columns if 'report_date' in c or 'as of date' in c][0]
        col_map[date_col] = 'Data'
        
        # Non-Commercial Long
        nc_long_col = [c for c in df_raw.columns if 'noncomm' in c and 'long' in c and 'all' in c][0]
        col_map[nc_long_col] = 'NC_Long'
        
        # Non-Commercial Short
        nc_short_col = [c for c in df_raw.columns if 'noncomm' in c and 'short' in c and 'all' in c][0]
        col_map[nc_short_col] = 'NC_Short'
        
        # Commercial Long (Escludendo tassativamente la stringa 'non')
        comm_long_col = [c for c in df_raw.columns if 'comm' in c and 'long' in c and 'all' in c and 'non' not in c][0]
        col_map[comm_long_col] = 'Comm_Long'
        
        # Commercial Short (Escludendo tassativamente la stringa 'non')
        comm_short_col = [c for c in df_raw.columns if 'comm' in c and 'short' in c and 'all' in c and 'non' not in c][0]
        col_map[comm_short_col] = 'Comm_Short'
        
    except IndexError:
        raise KeyError(f"🚨 ERRORE MATRICE: Schema CFTC non riconosciuto. Colonne trovate: {list(df_raw.columns)}")
        
    # Applicazione mapping dinamico
    df_clean = df_raw.rename(columns=col_map)
    
    # Pulizia Serie Storica e Asset
    df_clean['Data'] = pd.to_datetime(df_clean['Data'], errors='coerce')
    df_clean = df_clean.sort_values('Data').dropna(subset=['Data'])
    
    df_clean['Asset'] = df_clean['Asset_Raw'].str.strip().map(CFTC_MAPPING)
    df_clean = df_clean.dropna(subset=['Asset'])
    
    # Conversione numerica sicura e calcolo Posizionamenti Netti
    for col in ['NC_Long', 'NC_Short', 'Comm_Long', 'Comm_Short']:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
        
    df_clean['Net_NC'] = df_clean['NC_Long'] - df_clean['NC_Short']
    df_clean['Net_Comm'] = df_clean['Comm_Long'] - df_clean['Comm_Short']
    
    df_final = df_clean[['Data', 'Asset', 'Net_NC', 'Net_Comm']]
    
    output_path = os.path.join(base_dir, "cot_history.csv")
    df_final.to_csv(output_path, index=False)
    
    print(f"\n✅ MATRICE COT COMPLETATA E NORMALIZZATA.")
    print(f"-> Vettori processati: {len(df_final)}")
    print(f"-> File generato: '{output_path}'")
    print("\nAZIONE RICHIESTA: Fai il commit e il push del file 'cot_history.csv' verso GitHub.")

if __name__ == "__main__":
    build_offline_database()
    