import os
import pandas as pd
import numpy as np

# Mappatura estesa: Collega i nomi oscuri della CFTC ai nomi puliti per la Dashboard
CFTC_MAPPING = {
    # --- INDICI AZIONARI ---
    "E-MINI S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE": "S&P 500",
    "E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE": "S&P 500",
    "NASDAQ-100 STOCK INDEX (MINI) - CHICAGO MERCANTILE EXCHANGE": "Nasdaq 100",
    "NASDAQ MINI - CHICAGO MERCANTILE EXCHANGE": "Nasdaq 100",
    "DJIA x $5 - CHICAGO BOARD OF TRADE": "Dow Jones",
    "DOW JONES INDUSTRIAL AVG- x $5 - CHICAGO BOARD OF TRADE": "Dow Jones",
    "RUSSELL 2000 MINI INDEX FUTURE - ICE FUTURES U.S.": "Russell 2000",
    "RUSSELL E-MINI - CHICAGO MERCANTILE EXCHANGE": "Russell 2000",
    "NIKKEI STOCK AVERAGE - CHICAGO MERCANTILE EXCHANGE": "Nikkei 225",
    "NIKKEI STOCK AVERAGE YEN DENOM - CHICAGO MERCANTILE EXCHANGE": "Nikkei 225",
    "VIX FUTURES - CBOE FUTURES EXCHANGE": "VIX",

    # --- VALUTE E CRYPTO ---
    "EURO FX - CHICAGO MERCANTILE EXCHANGE": "EUR",
    "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE": "JPY",
    "BRITISH POUND STERLING - CHICAGO MERCANTILE EXCHANGE": "GBP",
    "BRITISH POUND - CHICAGO MERCANTILE EXCHANGE": "GBP",
    "CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE": "CAD",
    "AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE": "AUD",
    "SWISS FRANC - CHICAGO MERCANTILE EXCHANGE": "CHF",
    "NEW ZEALAND DOLLAR - CHICAGO MERCANTILE EXCHANGE": "NZD",
    "U.S. DOLLAR INDEX - ICE FUTURES U.S.": "USD Index",
    "USD INDEX - ICE FUTURES U.S.": "USD Index",
    "BITCOIN - CHICAGO MERCANTILE EXCHANGE": "Bitcoin",
    "MICRO BITCOIN - CHICAGO MERCANTILE EXCHANGE": "Micro Bitcoin",

    # --- METALLI PREZIOSI E INDUSTRIALI ---
    "GOLD - COMMODITY EXCHANGE INC.": "Gold",
    "SILVER - COMMODITY EXCHANGE INC.": "Silver",
    "PALLADIUM - NEW YORK MERCANTILE EXCHANGE": "Palladium",
    "PLATINUM - NEW YORK MERCANTILE EXCHANGE": "Platinum",
    "COPPER-GRADE #1 - COMMODITY EXCHANGE INC.": "Copper",
    "COPPER- #1 - COMMODITY EXCHANGE INC.": "Copper",

    # --- ENERGIA ---
    "CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE": "WTI Crude Oil",
    "WTI-PHYSICAL - NEW YORK MERCANTILE EXCHANGE": "WTI Crude Oil",
    "BRENT CRUDE OIL LAST DAY - NEW YORK MERCANTILE EXCHANGE": "Brent Crude",
    "CRUDE OIL, LIGHT SWEET-WTI - ICE FUTURES EUROPE": "WTI Crude (ICE)",
    "NATURAL GAS - NEW YORK MERCANTILE EXCHANGE": "Natural Gas",
    "RBOB GASOLINE - NEW YORK MERCANTILE EXCHANGE": "Gasoline",
    "HEATING OIL, N.Y. HARBOR - NEW YORK MERCANTILE EXCHANGE": "Heating Oil",

    # --- TASSI DI INTERESSE ---
    "10-YEAR U.S. TREASURY NOTES - CHICAGO BOARD OF TRADE": "10Y UST",
    "UST 10Y NOTE - CHICAGO BOARD OF TRADE": "10Y UST",
    "2-YEAR U.S. TREASURY NOTES - CHICAGO BOARD OF TRADE": "2Y UST",
    "2Y NOTE - CHICAGO BOARD OF TRADE": "2Y UST",
    "5-YEAR U.S. TREASURY NOTES - CHICAGO BOARD OF TRADE": "5Y UST",
    "U.S. TREASURY BONDS - CHICAGO BOARD OF TRADE": "UST Bonds",

    # --- AGRICOLI ---
    "CORN - CHICAGO BOARD OF TRADE": "Corn",
    "WHEAT-SRW - CHICAGO BOARD OF TRADE": "Wheat",
    "SOYBEANS - CHICAGO BOARD OF TRADE": "Soybeans",
    "COCOA - ICE FUTURES U.S.": "Cocoa",
    "COFFEE C - ICE FUTURES U.S.": "Coffee",
    "SUGAR NO. 11 - ICE FUTURES U.S.": "Sugar",
    "COTTON NO. 2 - ICE FUTURES U.S.": "Cotton",
    "LIVE CATTLE - CHICAGO MERCANTILE EXCHANGE": "Live Cattle"
}

def fetch_cftc_data(years_back: int = 4) -> pd.DataFrame:
    """
    Legge la matrice COT prioritaria dal database locale 'cot_history.csv' 
    Applicando pulizia date e mappatura asset automatica.
    """
    csv_path = "cot_history.csv"
    
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, low_memory=False)
            if df.empty:
                raise ValueError("File vuoto.")

            # 1. FIX DATE "1970-01-01": Decodifica i formati raw CFTC numerici
            if 'Report_Date_as_MM_DD_YYYY' in df.columns:
                df['Data'] = pd.to_datetime(df['Report_Date_as_MM_DD_YYYY'], errors='coerce')
            elif 'As_of_Date_In_Form_YYMMDD' in df.columns:
                df['Data'] = pd.to_datetime(df['As_of_Date_In_Form_YYMMDD'].astype(str).str.zfill(6), format='%y%m%d', errors='coerce')
            elif 'Data' in df.columns:
                # Se la colonna Data esiste ma contiene numeri (causa del 1970)
                if pd.api.types.is_numeric_dtype(df['Data']):
                    date_str = df['Data'].astype(str).str.split('.').str[0]
                    df['Data'] = pd.to_datetime(date_str, format='%Y%m%d', errors='coerce').fillna(
                                 pd.to_datetime(date_str, format='%y%m%d', errors='coerce'))
                else:
                    df['Data'] = pd.to_datetime(df['Data'], errors='coerce')

            df = df.dropna(subset=['Data'])

            # 2. APPLICAZIONE MAPPA ASSET (Mancante nel codice originale)
            if 'Market_and_Exchange_Names' in df.columns:
                df['Asset'] = df['Market_and_Exchange_Names'].map(CFTC_MAPPING)
            elif 'Asset' in df.columns:
                # Mappa i nomi grezzi se presenti nella colonna Asset
                mapped = df['Asset'].map(CFTC_MAPPING)
                df['Asset'] = mapped.combine_first(df['Asset'])
                
            # Filtro per mantenere SOLO gli asset che abbiamo mappato
            valid_assets = list(set(CFTC_MAPPING.values()))
            df = df[df['Asset'].isin(valid_assets)]

            # 3. NORMALIZZAZIONE NOMI COLONNE
            col_mappings = {
                'NonComm_Positions_Long_All': 'NC_Long',
                'NonComm_Positions_Short_All': 'NC_Short',
                'Comm_Positions_Long_All': 'Comm_Long',
                'Comm_Positions_Short_All': 'Comm_Short'
            }
            df = df.rename(columns=col_mappings)

            if 'Net_NC' not in df.columns and 'NC_Long' in df.columns and 'NC_Short' in df.columns:
                df['Net_NC'] = df['NC_Long'] - df['NC_Short']
            if 'Net_Comm' not in df.columns and 'Comm_Long' in df.columns and 'Comm_Short' in df.columns:
                df['Net_Comm'] = df['Comm_Long'] - df['Comm_Short']

            return df
        except Exception as e:
            raise ValueError(f"Errore nella lettura del file COT: {e}")
            
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
    
    # Pulizia Date per visualizzazione tabella
    latest['Data'] = latest['Data'].dt.strftime('%Y-%m-%d')
    latest['Z_NC'] = latest['Z_NC'].round(2)
    latest['Z_Comm'] = latest['Z_Comm'].round(2)
    
    latest['Alert_NC'] = np.where(latest['Z_NC'].abs() >= 1.8, "⭐", "")
    latest['Alert_Comm'] = np.where(latest['Z_Comm'].abs() >= 1.8, "⭐", "") if 'Z_Comm' in latest.columns else ""
    
    cols_to_return = [col for col in ['Data', 'Asset', 'Net_NC', 'Z_NC', 'Alert_NC', 'Net_Comm', 'Z_Comm', 'Alert_Comm'] if col in latest.columns]
    
    sort_col = 'Z_Comm' if 'Z_Comm' in latest.columns else 'Z_NC'
    latest = latest[cols_to_return].sort_values(by=sort_col, key=abs, ascending=False).reset_index(drop=True)
    
    return latest
