import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from backend_cot import fetch_cftc_data, calculate_cot_zscores
from backend_volume import fetch_ohlcv_data, calculate_volume_profile

# Cache a 24h per il COT: i report escono settimanalmente. 
# Impedisce il ban IP dai server della CFTC.
@st.cache_data(ttl=86400, show_spinner=False)
def load_and_process_cot():
    df_raw = fetch_cftc_data(years_back=4)
    return calculate_cot_zscores(df_raw, window_weeks=156)

def render_page2():
    st.markdown("""
    <style>
        .stApp { background-color: #0b1121 !important; color: #f8fafc !important; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
        hr { border-color: #1e293b !important; margin-top: 2rem !important; margin-bottom: 2rem !important; }
        .stDataFrame { border-radius: 8px; overflow: hidden; border: 1px solid #334155; }
        [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 1.7rem !important; font-weight: 800 !important; }
        [data-testid="stMetricLabel"] { color: #cbd5e1 !important; font-weight: 700 !important; font-size: 0.85rem !important; text-transform: uppercase !important; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🏛️ Volumi, COT & Z-Score")
    st.caption("Motore Vettoriale | Fonte: CFTC & OHLCV Exchange Data")

    # Separazione compartimenti analitici
    tab_cot, tab_vol = st.tabs(["📊 Analisi COT (Istituzionali)", "🎯 Point of Control (Volumi)"])

    # ==========================================
    # TAB 1: COMMITMENT OF TRADERS (Z-SCORE)
    # ==========================================
    with tab_cot:
        with st.sidebar:
            st.header("🔄 Sincronizzazione COT")
            if st.button("SCARICA REPORT CFTC", use_container_width=True):
                with st.spinner("Estrazione ZIP e parsing dai server CFTC..."):
                    st.cache_data.clear()
                    load_and_process_cot()
                st.rerun()

        try:
            with st.spinner("Calcolo vettori Z-Score a 156 settimane..."):
                df_cot = load_and_process_cot()
        except Exception as e:
            st.error(f"🚨 ERRORE DI RETE/PARSING CFTC: {e}")
            df_cot = pd.DataFrame()

        if not df_cot.empty:
            df_display = df_cot.copy()
            df_display['Data'] = df_display['Data'].dt.strftime('%Y-%m-%d')
            df_display['Net_NC'] = df_display['Net_NC'].apply(lambda x: f"{int(x):,}".replace(",", "."))
            df_display['Net_Comm'] = df_display['Net_Comm'].apply(lambda x: f"{int(x):,}".replace(",", "."))

            df_extremes = df_display[(df_display['Alert_NC'] == '⭐') | (df_display['Alert_Comm'] == '⭐')]

            st.markdown("### 🚨 Anomalie di Posizionamento Istituzionale")
            if not df_extremes.empty:
                st.markdown("""
                <div style="background-color: #450a0a; border-left: 5px solid #ef4444; padding: 15px; border-radius: 4px; color: #fca5a5; margin-bottom: 20px;">
                    <b>ATTENZIONE:</b> Scostamento critico rilevato (> 1.8 σ rispetto alla media mobile a 3 anni). 
                    Probabilità matematica di inversione mean-reverting elevata.
                </div>
                """, unsafe_allow_html=True)
                st.dataframe(df_extremes, use_container_width=True, hide_index=True)
            else:
                st.markdown("""
                <div style="background-color: #064e3b; border-left: 5px solid #10b981; padding: 15px; border-radius: 4px; color: #a7f3d0; margin-bottom: 20px;">
                    <b>NESSUNA ANOMALIA:</b> Attualmente nessun asset monitorato presenta squilibri oltre la soglia critica di ±1.8 σ.
                </div>
                """, unsafe_allow_html=True)

            st.divider()

            st.markdown("### 📋 Tabella Master COT (Completa)")
            st.dataframe(df_display, use_container_width=True, hide_index=True)

    # ==========================================
    # TAB 2: VOLUME PROFILE & P.O.C.
    # ==========================================
    with tab_vol:
        st.markdown("### Costruzione Profilo Volumetrico e Ricerca Nodo Primario")
        
        col_input, col_chart = st.columns([1, 3])
        
        with col_input:
            st.markdown("<br>", unsafe_allow_html=True)
            ticker = st.text_input("Ticker Asset (es. SPY, GLD, TLT)", value="SPY").upper()
            period = st.selectbox("Finestra Temporale", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=2)
            bins = st.slider("Risoluzione Livelli (Bins)", min_value=50, max_value=200, value=100, step=10)
            st.write("")
            esegui_poc = st.button("CALCOLA P.O.C.", type="primary", use_container_width=True)
            
        with col_chart:
            if esegui_poc:
                with st.spinner(f"Elaborazione tensori volumetrici (High-Low) per {ticker}..."):
                    df_ohlcv = fetch_ohlcv_data(ticker, period)
                    
                    if df_ohlcv.empty:
                        st.error(f"🚨 Dati OHLCV non disponibili per il ticker {ticker}. Verificare il simbolo o la connessione.")
                    else:
                        poc_price, df_vp = calculate_volume_profile(df_ohlcv, num_bins=bins)
                        
                        m1, m2 = st.columns(2)
                        m1.metric("POINT OF CONTROL (POC)", f"{poc_price:.2f}")
                        m2.metric("Candele OHLCV elaborate", len(df_ohlcv))
                        
                        # Rendering vettoriale del Volume Profile
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=df_vp['Volume'], 
                            y=df_vp['Price'], 
                            orientation='h',
                            marker_color='#0f766e',
                            name='Volume Scambiato'
                        ))
                        
                        # Tracciamento linea POC
                        fig.add_hline(
                            y=poc_price, 
                            line_dash="dash", 
                            line_color="#ef4444", 
                            line_width=2,
                            annotation_text=f"POC: {poc_price:.2f}",
                            annotation_position="bottom right",
                            annotation_font_color="#ef4444",
                            annotation_font_size=12
                        )
                        
                        fig.update_layout(
                            template="plotly_dark",
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            margin=dict(l=0, r=0, t=30, b=0),
                            xaxis_title="Contratti / Volume",
                            yaxis_title="Livelli di Prezzo",
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("👈 Inserisci i parametri dell'asset e avvia il calcolo.")
