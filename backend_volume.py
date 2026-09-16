import pandas as pd
import numpy as np
import yfinance as yf

# ==========================================================
# FASE 1: ESTRAZIONE DATI OHLCV (Regola 1 - Tolleranza Zero)
# ==========================================================
def fetch_ohlcv_data(ticker, period="1y"):
    """
    Estrae i dati Open, High, Low, Close e Volume reali.
    Nessun dato fittizio viene generato in caso di errore.
    """
    try:
        data = yf.download(ticker, period=period, progress=False)
        if data.empty:
            return pd.DataFrame()

        # Gestione compatibilità MultiIndex per le versioni recenti di yfinance
        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.get_level_values(0):
                data = data.copy()
                data.columns = data.columns.get_level_values(0)
            else:
                data.columns = [col[0] for col in data.columns]

        # Validazione colonne necessarie
        req_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in req_cols):
            return pd.DataFrame()

        df = data[req_cols].copy()
        
        # Filtro anomalie: rimuove giorni senza volumi scambiati
        df = df[df['Volume'] > 0].dropna()
        return df
        
    except Exception as e:
        print(f"🚨 ERRORE API YFINANCE OHLCV ({ticker}): {e}")
        return pd.DataFrame()

# ==========================================================
# FASE 2: MOTORE MATEMATICO VOLUME PROFILE E P.O.C. (Regola 2)
# ==========================================================
def calculate_volume_profile(df, num_bins=100):
    """
    Distribuisce il volume sull'intera estensione della candela (High-Low).
    Calcola il POC (Point of Control) esatto e il Profilo Volumetrico.
    """
    if df.empty or len(df) < 5:
        return np.nan, pd.DataFrame()

    min_price = df['Low'].min()
    max_price = df['High'].max()

    # Prevenzione divisione per zero in caso di asset flat
    if min_price == max_price:
        return min_price, pd.DataFrame({'Price': [min_price], 'Volume': [df['Volume'].sum()]})

    # Creazione della matrice dei livelli di prezzo (bins)
    tick_size = (max_price - min_price) / num_bins
    bins = np.arange(min_price, max_price + tick_size, tick_size)
    profile = np.zeros(len(bins) - 1)

    # Distribuzione vettoriale iterativa
    for _, row in df.iterrows():
        high = row['High']
        low = row['Low']
        vol = float(row['Volume'])

        if high == low:
            bin_idx = np.digitize(high, bins) - 1
            if 0 <= bin_idx < len(profile):
                profile[bin_idx] += vol
            continue

        # Mappatura dei livelli di prezzo attraversati dalla candela
        bin_start = np.digitize(low, bins) - 1
        bin_end = np.digitize(high, bins) - 1

        # Limiti di sicurezza array
        bin_start = max(0, bin_start)
        bin_end = min(len(profile) - 1, bin_end)

        # Assegnazione proporzionale del volume
        if bin_start == bin_end:
            profile[bin_start] += vol
        else:
            n_bins = bin_end - bin_start + 1
            vol_per_bin = vol / n_bins
            profile[bin_start:bin_end+1] += vol_per_bin

    # Strutturazione dei risultati
    df_profile = pd.DataFrame({
        'Price': bins[:-1] + (tick_size / 2.0), # Prezzo centrale del livello
        'Volume': profile
    })

    # P.O.C. = Livello con la massima concentrazione di volume scambiato
    poc_idx = df_profile['Volume'].idxmax()
    poc_price = df_profile.loc[poc_idx, 'Price']

    return poc_price, df_profile

# ==========================================================
# ESECUZIONE DI TEST E CONVALIDA MATEMATICA
# ==========================================================
if __name__ == "__main__":
    test_ticker = "SPY"
    print(f"Estrazione flussi OHLCV per {test_ticker}...")
    
    df_test = fetch_ohlcv_data(test_ticker, period="6mo")
    
    if not df_test.empty:
        poc, vp_data = calculate_volume_profile(df_test)
        
        print("\n=== CONVALIDA MOTORE VOLUMI ===")
        print(f"Candele elaborate: {len(df_test)}")
        print(f"POINT OF CONTROL (POC): {poc:.2f}")
        
        print("\nTOP 5 NODI AD ALTO VOLUME (HVN):")
        top_hvn = vp_data.sort_values(by='Volume', ascending=False).head(5)
        top_hvn['Volume'] = top_hvn['Volume'].apply(lambda x: f"{int(x):,}".replace(",", "."))
        print(top_hvn.to_string(index=False))
    else:
        print("🚨 Test fallito. Dati OHLCV non disponibili.")
