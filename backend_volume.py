import pandas as pd
import yfinance as yf

def fetch_ohlcv_data(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """
    Estrae i dati storici reali da Yahoo Finance (Regola 1).
    Gestisce la struttura MultiIndex e restituisce un DataFrame pulito.
    """
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            return pd.DataFrame()
        
        # Flatten MultiIndex se presente (comportamento standard di yfinance recente)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.reset_index()
        return df
    except Exception:
        return pd.DataFrame()

def calculate_volume_profile(df: pd.DataFrame, num_bins: int = 100):
    """
    Costruisce il Volume Profile e identifica il Point of Control (POC).
    Regola 2: Frazionamento geometrico del volume sull'escursione High-Low.
    """
    # Controlli di integrità
    if df.empty or not all(col in df.columns for col in ['High', 'Low', 'Volume']):
        return 0.0, pd.DataFrame(columns=['Price', 'Volume'])
    
    # Conversione forzata a float per evitare errori di calcolo su int
    df['High'] = df['High'].astype(float)
    df['Low'] = df['Low'].astype(float)
    df['Volume'] = df['Volume'].astype(float)
    
    price_min = df['Low'].min()
    price_max = df['High'].max()
    
    # Gestione edge case (Asset piatto o dataset di un solo giorno senza escursione)
    if price_min == price_max:
        return price_min, pd.DataFrame({'Price': [price_min], 'Volume': [df['Volume'].sum()]})
        
    bin_size = (price_max - price_min) / num_bins
    bins_edges = [price_min + i * bin_size for i in range(num_bins + 1)]
    
    # Inizializzazione profilo
    profile = {i: 0.0 for i in range(num_bins)}
    
    # Ripartizione volumetrica proporzionale (Regola 2)
    for _, row in df.iterrows():
        h = row['High']
        l = row['Low']
        v = row['Volume']
        
        if h == l:
            # Candela doji estrema: assegnazione intera al bin corrispondente
            for i in range(num_bins):
                if bins_edges[i] <= h <= bins_edges[i+1]:
                    profile[i] += v
                    break
            continue
            
        range_span = h - l
        
        for i in range(num_bins):
            b_low = bins_edges[i]
            b_high = bins_edges[i+1]
            
            # Calcolo intersezione tra candela e bin corrente
            overlap_low = max(l, b_low)
            overlap_high = min(h, b_high)
            
            if overlap_high > overlap_low:
                fraction = (overlap_high - overlap_low) / range_span
                profile[i] += v * fraction
                
    # Costruzione DataFrame finale compatibile con il frontend
    profile_df = pd.DataFrame({
        'Price': [(bins_edges[i] + bins_edges[i+1]) / 2 for i in range(num_bins)],
        'Volume': list(profile.values())
    })
    
    # Identificazione del Point of Control (POC)
    poc_idx = profile_df['Volume'].idxmax()
    poc_price = profile_df.loc[poc_idx, 'Price']
    
    return poc_price, profile_df
