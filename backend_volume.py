import pandas as pd
import yfinance as yf

def fetch_ohlcv_data(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        return df
    except Exception as e:
        return pd.DataFrame()

def calculate_volume_profile(df: pd.DataFrame, bins: int = 20) -> pd.DataFrame:
    if df.empty or 'High' not in df.columns or 'Low' not in df.columns or 'Volume' not in df.columns:
        return pd.DataFrame()
    
    price_min = df['Low'].min()
    price_max = df['High'].max()
    
    if price_min == price_max:
        return pd.DataFrame()
        
    bin_size = (price_max - price_min) / bins
    bins_edges = [price_min + i * bin_size for i in range(bins + 1)]
    
    profile = {i: 0.0 for i in range(bins)}
    
    for _, row in df.iterrows():
        h = row['High']
        l = row['Low']
        v = row['Volume']
        
        if h == l:
            continue
            
        range_span = h - l
        for i in range(bins):
            b_low = bins_edges[i]
            b_high = bins_edges[i+1]
            
            overlap_low = max(l, b_low)
            overlap_high = min(h, b_high)
            
            if overlap_high > overlap_low:
                fraction = (overlap_high - overlap_low) / range_span
                profile[i] += v * fraction
                
    profile_df = pd.DataFrame({
        'Price_Level': [(bins_edges[i] + bins_edges[i+1]) / 2 for i in range(bins)],
        'Volume': list(profile.values())
    })
    
    return profile_df
