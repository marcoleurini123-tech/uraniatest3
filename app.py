import streamlit as st
from pages_modules.page1_macro import render_page1

# 1. Configurazione Globale dell'Infrastruttura
st.set_page_config(
    page_title="Urania - Macro Intelligence Terminal",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Iniezione CSS Corretta (Header visibile per permettere il toggle della sidebar)
st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 0rem; }
        footer { visibility: hidden; }
        #MainMenu { visibility: hidden; }
        [data-testid="stMetricValue"] { font-size: 1.8rem; }
    </style>
""", unsafe_allow_html=True)

def main():
    # 3. Router Operativo Istituzionale
    st.sidebar.title("🌌 Sistema Urania")
    st.sidebar.caption("Analisi quantitativa. Esecuzione meccanica.")
    
    st.sidebar.markdown("---")
    
    modulo_attivo = st.sidebar.radio(
        "SELEZIONE MODULO OPERATIVO",
        [
            "1. Macro & Institutional EOD",
            "2. Volumi, COT & Z-Score",
            "3. Strutture Derivati & Coperture"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("Status: **Online** | Motore: **Vettoriale**")

    # 4. Assegnazione Compartimenti
    if modulo_attivo == "1. Macro & Institutional EOD":
        render_page1()
        
    elif modulo_attivo == "2. Volumi, COT & Z-Score":
        st.title("2. Analisi Flussi, COT e Stagionalità")
        st.warning("Modulo in attesa di implementazione matematica (Regola 4).")
        
    elif modulo_attivo == "3. Strutture Derivati & Coperture":
        st.title("3. Gestione Coperture e Opzioni")
        st.warning("Modulo in attesa di implementazione matematica (Regola 4).")

if __name__ == "__main__":
    main()
