import streamlit as st
import os
from PIL import Image

# ==========================================
# CONFIGURAZIONE PAGINA (Deve essere la prima istruzione)
# ==========================================
st.set_page_config(
    page_title="Urania - Macro Intelligence Terminal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Importiamo i moduli delle pagine dalla cartella
from pages_modules.page1_macro import render_page1
from pages_modules.page2_cot import render_page2

def verify_access():
    """
    Funzione per gestire la grafica della sidebar (logo)
    e un eventuale sistema di accesso.
    Include l'Anti-Crash per le immagini.
    """
    logo_path = "logo.png"
    
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    
    if os.path.exists(logo_path):
        try:
            # Tenta di aprire e caricare l'immagine
            img = Image.open(logo_path)
            # Usa use_container_width=True come richiesto dalle nuove versioni di Streamlit
            st.sidebar.image(img, use_container_width=True)
        except Exception:
            # Se il logo è corrotto o irriconoscibile, stampa solo il titolo senza far crashare l'app
            st.sidebar.markdown("<h2 style='text-align: center; color: #cbd5e1;'>Urania Terminal</h2>", unsafe_allow_html=True)
    else:
        # Se il file logo.png non esiste affatto
        st.sidebar.markdown("<h2 style='text-align: center; color: #cbd5e1;'>Urania Terminal</h2>", unsafe_allow_html=True)

    st.sidebar.markdown("<br>", unsafe_allow_html=True)

def main():
    # 1. Carica il logo / sidebar header
    verify_access()
    
    # 2. Informazioni di base sulla Sidebar
    st.sidebar.title("🌌 Sistema Urania")
    st.sidebar.caption("Analisi quantitativa. Esecuzione meccanica.")
    st.sidebar.markdown("---")
    st.sidebar.markdown("Status: **Online** | Motore: **Vettoriale**")
    st.sidebar.markdown("---")
    
    # 3. Menu di navigazione tra i vari moduli
    modulo_attivo = st.sidebar.radio(
        "SELEZIONE MODULO OPERATIVO",
        (
            "1. Macro & Institutional EOD", 
            "2. Volumi, COT & Z-Score",
            "3. Strutture Derivati & Coperture"
        )
    )
    
    # 4. Routing: chiama la pagina giusta in base a cosa clicchi
    if modulo_attivo == "1. Macro & Institutional EOD":
        render_page1()
    elif modulo_attivo == "2. Volumi, COT & Z-Score":
        render_page2()
    elif modulo_attivo == "3. Strutture Derivati & Coperture":
        st.title("3. Gestione Coperture e Opzioni")
        st.warning("Modulo in attesa di implementazione matematica (Regime di Transizione).")

if __name__ == "__main__":
    main()
