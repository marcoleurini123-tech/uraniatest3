import streamlit as st
import pandas as pd
from backend_cot import fetch_cftc_data, calculate_cot_zscores

# Cache a 24 ore: i report COT escono settimanalmente (venerdì). 
# Ripetere chiamate multiple bloccherebbe l'IP da parte della CFTC.
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
    </style>
    """, unsafe_allow_html=True)

    st.title("🏛️ Analisi COT (Commitment of Traders)")
    st.caption("Fonte: CFTC | Motore: Z-Score 156 Settimane | Aggiornamento: Settimanale")

    with st.sidebar:
        st.header("🔄 Sincronizzazione CFTC")
        if st.button("SCARICA REPORT UFFICIALE", use_container_width=True):
            with st.spinner("Estrazione ZIP e parsing dai server CFTC..."):
                st.cache_data.clear()
                load_and_process_cot()
            st.success("Database COT aggiornato.")
            st.rerun()

    try:
        with st.spinner("Calcolo vettori Z-Score e deviazioni standard..."):
            df_cot = load_and_process_cot()
    except Exception as e:
        st.error(f"🚨 ERRORE DI RETE/PARSING CFTC: {e}")
        return

    if df_cot.empty:
        st.warning("⚠️ Database COT vuoto o dati insufficienti per il calcolo a 156 settimane.")
        return

    # Formattazione per la UI
    df_display = df_cot.copy()
    df_display['Data'] = df_display['Data'].dt.strftime('%Y-%m-%d')
    df_display['Net_NC'] = df_display['Net_NC'].apply(lambda x: f"{int(x):,}".replace(",", "."))
    df_display['Net_Comm'] = df_display['Net_Comm'].apply(lambda x: f"{int(x):,}".replace(",", "."))

    # Filtro logico per gli estremi richiesti (Z-Score > 1.8 o < -1.8)
    df_extremes = df_display[(df_display['Alert_NC'] == '⭐') | (df_display['Alert_Comm'] == '⭐')]

    st.markdown("### 🚨 Anomalie di Posizionamento Istituzionale")
    if not df_extremes.empty:
        st.markdown("""
        <div style="background-color: #450a0a; border-left: 5px solid #ef4444; padding: 15px; border-radius: 4px; color: #fca5a5; margin-bottom: 20px;">
            <b>ATTENZIONE:</b> I seguenti asset registrano uno scostamento superiore a 1.8 deviazioni standard rispetto alla media mobile a 3 anni. Rischio inversione mean-reverting elevato.
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

    st.markdown("### 📊 Tabella Master COT (Completa)")
    st.dataframe(df_display, use_container_width=True, hide_index=True)
