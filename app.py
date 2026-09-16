import streamlit as st
import os
from pages_modules.page1_macro import render_page1
from pages_modules.page2_cot import render_page2

st.set_page_config(page_title="Urania - Macro Intelligence Terminal", page_icon="🌌", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 0rem; }
        footer { visibility: hidden; }
        #MainMenu { visibility: hidden; }
        [data-testid="stMetricValue"] { font-size: 1.8rem; }
    </style>
""", unsafe_allow_html=True)

def verify_access():
    """Regola 3: Controllo stringente della chiave di sicurezza."""
    if "APP_PASSWORD" not in st.secrets:
        st.error("🚨 ERRORE DI SICUREZZA: 'APP_PASSWORD' non configurata in st.secrets.")
        st.stop()

    if "auth_status" not in st.session_state:
        st.session_state["auth_status"] = False

    if not st.session_state["auth_status"]:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            logo_path = "logo.png"
            if os.path.exists(logo_path):
                st.image(logo_path, use_column_width=True)
            else:
                st.markdown("<h2 style='text-align: center; color: #cbd5e1;'>🔒 URANIA SYSTEM</h2>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            pwd_input = st.text_input("Inserire Token di Decrittazione:", type="password", key="pwd_input")
            
            if st.button("AUTENTICAZIONE", use_container_width=True):
                if pwd_input == st.secrets["APP_PASSWORD"]:
                    st.session_state["auth_status"] = True
                    st.rerun()
                else:
                    st.error("🚨 Accesso Negato. Token non valido.")
        st.stop()

def main():
    verify_access()

    st.sidebar.title("🌌 Sistema Urania")
    st.sidebar.caption("Analisi quantitativa. Esecuzione meccanica.")
    st.sidebar.markdown("---")
    
    modulo_attivo = st.sidebar.radio(
        "SELEZIONE MODULO OPERATIVO",
        ["1. Macro & Institutional EOD", "2. Volumi, COT & Z-Score", "3. Strutture Derivati & Coperture"]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("Status: **Online** | Motore: **Vettoriale** | Rete: **Sicura**")

    if modulo_attivo == "1. Macro & Institutional EOD":
        render_page1()
    elif modulo_attivo == "2. Volumi, COT & Z-Score":
        render_page2()
    elif modulo_attivo == "3. Strutture Derivati & Coperture":
        st.title("3. Gestione Coperture e Opzioni")
        st.warning("Modulo in attesa di implementazione matematica (Regola 4).")

if __name__ == "__main__":
    main()
