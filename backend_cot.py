import os
import pandas as pd
import numpy as np

def fetch_cftc_data(years_back: int = 4) -> pd.DataFrame:
    """
    Legge la matrice COT prioritaria dal database locale 'cot_history.csv' (Regola 1 e 4).
    Si aspetta il file formattato dall'updater offline: ['Data', 'Asset', 'Net_NC', 'Net_Comm']
    """
    csv_path = "cot_history.csv"
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError("IMPOSSIBILE CARICARE LA MATRICE COT: File 'cot_history.csv' assente nella root.")
        
    try:
        # Lettura bruta, senza inferenze automatiche pericolose
        df = pd.read_csv(csv_path, low_memory=False)
        
        if df.empty:
            raise ValueError("Il file 'cot_history.csv' è vuoto.")

        # Verifica conformità strutturale derivante dall'updater
        required_cols = ['Data', 'Asset', 'Net_NC', 'Net_Comm']
        if not all(col in df.columns for col in required_cols):
            raise KeyError(f"Struttura CSV non valida. Colonne attese: {required_cols}")

        # Parsing rigoroso della Data (Regola 1)
        df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
        df = df.dropna(subset=['Data', 'Asset'])
        
        # Filtraggio temporale matematico
        cutoff_date = pd.Timestamp.now() - pd.DateOffset(years=years_back)
        df = df[df['Data'] >= cutoff_date]
        
        return df

    except Exception as e:
        raise ValueError(f"Errore critico nella lettura o parsing del file COT: {str(e)}")


def calculate_cot_zscores(df_cot: pd.DataFrame, window_weeks: int = 156) -> pd.DataFrame:
    """
    Calcola lo Z-Score storico sulle Posizioni Nette (Regola 2).
    """
    if df_cot.empty:
        return pd.DataFrame()

    # Copia indipendente per non alterare il reference in cache
    df = df_cot.copy()
    
    # ORDINAMENTO CRITICO: I dati devono essere ordinati storicamente (crescente) per il calcolo rolling
    df = df.sort_values(by=['Asset', 'Data'])
    
    # Costante di smoothing per divisione per zero
    epsilon = 1e-9
    
    # ---------------------------------------------------------
    # Calcolo Z-Score Non-Commercial (Speculatori)
    # ---------------------------------------------------------
    df['NC_Mean'] = df.groupby('Asset')['Net_NC'].transform(lambda x: x.rolling(window_weeks, min_periods=window_weeks//2).mean())
    df['NC_Std'] = df.groupby('Asset')['Net_NC'].transform(lambda x: x.rolling(window_weeks, min_periods=window_weeks//2).std())
    df['Z_NC'] = (df['Net_NC'] - df['NC_Mean']) / (df['NC_Std'] + epsilon)
    
    # ---------------------------------------------------------
    # Calcolo Z-Score Commercial (Hedger)
    # ---------------------------------------------------------
    df['Comm_Mean'] = df.groupby('Asset')['Net_Comm'].transform(lambda x: x.rolling(window_weeks, min_periods=window_weeks//2).mean())
    df['Comm_Std'] = df.groupby('Asset')['Net_Comm'].transform(lambda x: x.rolling(window_weeks, min_periods=window_weeks//2).std())
    df['Z_Comm'] = (df['Net_Comm'] - df['Comm_Mean']) / (df['Comm_Std'] + epsilon)
    
    # Estrazione dell'ultimo snapshot disponibile per ogni asset (Dashboard View)
    latest = df.groupby('Asset').tail(1).copy()
    
    # Eliminazione asset senza storico sufficiente per lo Z-Score
    latest = latest.dropna(subset=['Z_NC', 'Z_Comm'])
    
    # ---------------------------------------------------------
    # Formattazione e Valutazione Outlier
    # ---------------------------------------------------------
    latest['Z_NC'] = latest['Z_NC'].round(2)
    latest['Z_Comm'] = latest['Z_Comm'].round(2)
    
    # Alert matematico a ±1.8 Deviazioni Standard
    latest['Alert_NC'] = np.where(latest['Z_NC'].abs() >= 1.8, "⭐", "")
    latest['Alert_Comm'] = np.where(latest['Z_Comm'].abs() >= 1.8, "⭐", "")
    
    # Selezione colonne finali per la UI
    cols_to_return = ['Data', 'Asset', 'Net_NC', 'Z_NC', 'Alert_NC', 'Net_Comm', 'Z_Comm', 'Alert_Comm']
    
    # Ordinamento per magnitudo assoluta dell'anomalia Hedger (Z_Comm)
    latest = latest[cols_to_return].sort_values(by='Z_Comm', key=abs, ascending=False).reset_index(drop=True)
    
    return latest
