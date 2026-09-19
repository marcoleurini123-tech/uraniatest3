import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from backend_engine import (
    load_db, save_db, fetch_yahoo_data, fetch_bridge_data, 
    fetch_squeezemetrics_data, fetch_cboe_pc_ratio, COLUMNS,
    fetch_regime_baskets_data, calculate_regime_matrix,
    calculate_macro_cycle_phase, calculate_risk_propensity, 
    evaluate_risk_override, sync_all_data
)

@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_regime_data():
    return fetch_regime_baskets_data(period="10y")

def render_page1():
    st.markdown("""
    <style>
        .stApp { background-color: #0b1121 !important; color: #f8fafc !important; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
        [data-testid="stMetricValue"], [data-testid="stMetricValue"] > div { color: #ffffff !important; font-size: 1.7rem !important; font-weight: 800 !important; }
        [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * { color: #cbd5e1 !important; font-weight: 700 !important; font-size: 0.85rem !important; text-transform: uppercase !important; letter-spacing: 0.5px !important; opacity: 1 !important; visibility: visible !important; }
        hr { border-color: #1e293b !important; margin-top: 2rem !important; margin-bottom: 2rem !important; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🛡️ Terminale Macro Professionale")
    st.caption("Status: Online | Motore: Vettoriale")
    
    df = load_db()

    with st.sidebar:
        st.header("⚙️ Override Dati EOD")
        st.caption("I dati immessi manualmente sovrascrivono il fetch API per la data selezionata.")
        
        with st.form("manual_entry"):
            m_date = st.date_input("Data Riferimento", datetime.now())
            m_v1 = st.number_input("VIX 1D", 0.0, format="%.2f")
            m_move = st.number_input("MOVE Index", 0.0, format="%.2f")
            m_pc = st.number_input("Put/Call Ratio", 0.0, format="%.2f")
            m_dix = st.number_input("DIX (%)", 0.0, format="%.1f")
            m_gex = st.number_input("GEX (Assoluto)", 0.0, format="%.0f")
            
            if st.form_submit_button("1. BLINDA DATI NEL DB"):
                dt = pd.to_datetime(m_date).normalize()
                if not df.empty and dt in df['Data'].values:
                    idx = df.index[df['Data'] == dt].tolist()[0]
                else:
                    new_row = {c: np.nan for c in COLUMNS}
                    new_row["Data"] = dt
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                    idx = df.index[-1]
                
                if m_v1 > 0: df.at[idx, 'VIX1D'] = m_v1
                if m_move > 0: df.at[idx, 'MOVE'] = m_move
                if m_pc > 0: df.at[idx, 'P_C'] = m_pc
                if m_dix > 0: df.at[idx, 'DIX'] = m_dix
                if m_gex != 0: df.at[idx, 'GEX'] = m_gex
                
                save_db(df)
                st.success(f"Sessione {dt.strftime('%Y-%m-%d')} registrata con rigore.")
                st.rerun()

        st.divider()
        st.header("🔄 Fetch Istituzionale")
        
        if st.button("2. SINCRONIZZA FLUSSI API", use_container_width=True):
            with st.spinner("Estrazione ed allineamento tensori temporali in corso..."):
                sync_all_data()
                st.cache_data.clear()
                st.rerun()

    if df.empty:
        st.warning("⚠️ Database locale vuoto. Procedere con l'inizializzazione dei flussi API.")
        return

    df = df.sort_values("Data").reset_index(drop=True)
    num_cols = [c for c in COLUMNS if c != "Data" and c in df.columns]
    df[num_cols] = df[num_cols].ffill(limit=7)

    if 'Net_Liquidity' in df.columns:
        df['Liq_Delta_5D'] = df['Net_Liquidity'].pct_change(periods=5) * 100
    df['Ratio_GO'] = np.where(df['USO'] > 0, df['GLD'] / df['USO'], np.nan) if 'USO' in df.columns and 'GLD' in df.columns else np.nan
    df['Ratio_Risk'] = np.where(df['XLP'] > 0, df['XLY'] / df['XLP'], np.nan) if 'XLP' in df.columns and 'XLY' in df.columns else np.nan
    df['Ratio_Br'] = np.where(df['RSP'] > 0, df['SPY'] / df['RSP'], np.nan) if 'RSP' in df.columns and 'SPY' in df.columns else np.nan

    last = df.iloc[-1]
    
    # ==========================================================
    # MODULO RISK MANAGEMENT & OPPORTUNITY
    # ==========================================================
    # QUI AVVENIVA IL CRASH: Ora le 5 variabili corrispondono perfettamente al backend.
    is_risk_off, risk_reasons, bullish_reasons, current_skew, current_vix = evaluate_risk_override(df)

    if is_risk_off:
        st.markdown(f"""
        <div style="background-color: #450a0a; border: 1px solid #ef4444; border-radius: 6px; padding: 16px; margin-bottom: 16px;">
            <h3 style="margin:0; color:#ef4444; font-size: 1.2rem;">🚨 HARD OVERRIDE ATTIVO: RISK OFF / PANICO</h3>
            <p style="margin-top:8px; color: #fca5a5; font-size: 0.95rem; font-weight:bold;">BLOCCO OPERATIVO ASSOLUTO.</p>
            <ul style="margin:0; padding-left:20px; color: #fca5a5; font-size: 0.95rem;">
                {''.join([f'<li>⚠️ {r}</li>' for r in risk_reasons])}
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    if bullish_reasons:
        st.markdown(f"""
        <div style="background-color: #064e3b; border: 1px solid #10b981; border-radius: 6px; padding: 16px; margin-bottom: 24px;">
            <h3 style="margin:0; color:#10b981; font-size: 1.2rem;">🟢 SEGNALE CONTRARIAN: ACCUMULO DARK POOL</h3>
            <ul style="margin-top:8px; padding-left:20px; color: #a7f3d0; font-size: 0.95rem;">
                {''.join([f'<li>📈 {r}</li>' for r in bullish_reasons])}
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # ==========================================================
    # KPI DASHBOARD EOD
    # ==========================================================
    r1 = st.columns(6)
    
    dix_v = last.get('DIX', np.nan)
    r1[0].metric("DIX", f"{dix_v:.1f}%" if not pd.isna(dix_v) else "N/A", "🟢 BULLISH" if dix_v > 45 else "⚪ NEUTRO")
    gex_v = last.get('GEX', np.nan)
    r1[1].metric("GEX", f"{gex_v:,.0f}" if not pd.isna(gex_v) else "N/A", "🔴 SQUEEZE" if gex_v < 0 else "🟢 STABILE", delta_color="inverse")
    pc_v = last.get('P_C', np.nan)
    pc_status = "🟢 PANICO" if pc_v > 1.05 else ("🔴 AVIDITÀ" if 0 < pc_v < 0.7 else "⚪ NEUTRO")
    r1[2].metric("P/C RATIO", f"{pc_v:.2f}" if not pd.isna(pc_v) else "N/A", pc_status)
    r1[3].metric("SKEW", f"{current_skew:.1f}" if not pd.isna(current_skew) else "N/A", "⚠️ BLACK SWAN" if current_skew >= 145 else "🟢 OK", delta_color="inverse")
    move_v = last.get('MOVE', np.nan)
    r1[4].metric("MOVE", f"{move_v:.1f}" if not pd.isna(move_v) else "N/A", "🔴 STRESS BOND" if move_v >= 115 else "🟢 CALMO", delta_color="inverse")
    liq_d = last.get('Liq_Delta_5D', np.nan)
    liq_col = "normal" if not pd.isna(liq_d) and liq_d >= 0 else "inverse"
    r1[5].metric("Δ LIQ. 5D", f"{liq_d:.2f}%" if not pd.isna(liq_d) else "N/A", "📉 CONTRAZIONE" if not pd.isna(liq_d) and liq_d < 0 else "📈 ESPANSIONE", delta_color=liq_col)

    st.write("") 
    r2 = st.columns(6)
    
    dxy_v = last.get('DXY', np.nan)
    r2[0].metric("DXY", f"{dxy_v:.2f}" if not pd.isna(dxy_v) else "N/A", "🔴 USD UP" if dxy_v > 103.5 else "🟢 USD DOWN", delta_color="inverse")
    rgo_v = last.get('Ratio_GO', np.nan)
    r2[1].metric("GOLD/OIL", f"{rgo_v:.2f}" if not pd.isna(rgo_v) else "N/A", "⚠️ ALERT" if rgo_v > 2.5 else "🟢 OK")
    tlt_v = last.get('TLT', np.nan)
    tlt_status = "📈 TASSI DOWN" if len(df) > 1 and tlt_v > df.iloc[-2].get('TLT', 0) else "📉 TASSI UP"
    r2[2].metric("TLT PRICE", f"${tlt_v:.2f}" if not pd.isna(tlt_v) else "N/A", tlt_status)
    rrisk_v = last.get('Ratio_Risk', np.nan)
    r2[3].metric("XLY/XLP", f"{rrisk_v:.2f}" if not pd.isna(rrisk_v) else "N/A", "🟢 RISK-ON" if rrisk_v > 1.45 else "🔴 DIFESA")
    rbr_v = last.get('Ratio_Br', np.nan)
    r2[4].metric("SPY/RSP", f"{rbr_v:.2f}" if not pd.isna(rbr_v) else "N/A", "⚠️ ALERT" if rbr_v > 3.5 else "🟢 SANA")
    v1d, vx = last.get('VIX1D', np.nan), last.get('VIX', np.nan)
    v_stat = "🔴 INVERTITA" if not pd.isna(v1d) and not pd.isna(vx) and v1d > vx else "🟢 CONTANGO"
    r2[5].metric("CURVA VIX", f"{v1d:.1f}/{vx:.1f}" if not pd.isna(v1d) and not pd.isna(vx) else "N/A", v_stat)

    st.divider()

    # ==========================================================
    # CARDS MACRO: REGIME E PROPENSIONE AL RISCHIO
    # ==========================================================
    with st.spinner("Computazione tensori statistici..."):
        df_regime_prices = get_cached_regime_data()
        df_matrix, dominant_regime, conf_pct = calculate_regime_matrix(df_regime_prices)
        risk_metrics, risk_err = calculate_risk_propensity(df)

    col_q1, col_q2 = st.columns(2)
    
    with col_q1:
        st.markdown(f"""
        <div style="background-color:#1e293b; padding:24px; border-radius:8px; border: 1px solid #334155;">
            <h4 style="color:#cbd5e1; margin-top:0; font-size:11px; font-weight: 700; text-transform:uppercase; letter-spacing: 1px;">Regime Economico Predominante</h4>
            <div style="display:flex; justify-content:space-between; align-items:flex-end; margin-top: 24px;">
                <div>
                    <span style="color:#ffffff; font-size:22px; font-weight:800; text-transform:uppercase;">{dominant_regime}</span>
                </div>
                <div style="text-align:right;">
                    <span style="color:#f59e0b; font-size:26px; font-weight:800;">{conf_pct}%</span>
                </div>
            </div>
            <div style="margin-top:20px; height:6px; width:100%; background: linear-gradient(90deg, #ef4444 0%, #f59e0b 50%, #10b981 100%); border-radius:4px;"></div>
        </div>
        """, unsafe_allow_html=True)

    with col_q2:
        if risk_err or not risk_metrics:
            r_status, r_on, r_off = "N/D", 0, 0
            r_color = "#94a3b8"
        else:
            r_status = risk_metrics['Status']
            r_on = risk_metrics['Risk_On_Pct']
            r_off = risk_metrics['Risk_Off_Pct']
            r_color = "#10b981" if r_status == "RISK ON" else ("#ef4444" if r_status == "RISK OFF" else "#f59e0b")
            
        st.markdown(f"""
        <div style="background-color:#1e293b; padding:24px; border-radius:8px; border: 1px solid #334155;">
            <h4 style="color:#cbd5e1; margin-top:0; font-size:11px; font-weight: 700; text-transform:uppercase; letter-spacing: 1px;">Propensione al Rischio (Z-Score Storico)</h4>
            <div style="display:flex; justify-content:space-between; align-items:flex-end; margin-top: 24px;">
                <div>
                    <span style="color:{r_color}; font-size:22px; font-weight:800; text-transform:uppercase;">{r_status}</span>
                </div>
                <div style="text-align:right;">
                    <span style="color:#10b981; font-size:26px; font-weight:800;">{r_on}%</span>
                </div>
            </div>
            <div style="margin-top:20px; display:flex; border-radius:4px; overflow:hidden; height:6px;">
                <div style="width:{r_off}%; background-color:#ef4444;"></div>
                <div style="width:{r_on}%; background-color:#10b981;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    # ==========================================================
    # MATRICE HEATMAP (NORMALIZZATA PER COLONNA + VINCITORE BIANCO)
    # ==========================================================
    st.markdown("### 🗺️ Matrice dei Regimi di Mercato")

    if not df_matrix.empty:
        df_numeric = df_matrix.apply(pd.to_numeric, errors='coerce')
        
        # Creiamo un dataframe vuoto per le coordinate dei colori
        df_norm = pd.DataFrame(index=df_numeric.index, columns=df_numeric.columns)
        
        # Normalizzazione COLONNA per COLONNA
        for col in df_numeric.columns:
            col_min = df_numeric[col].min()
            col_max = df_numeric[col].max()
            
            if col_max == col_min:
                df_norm[col] = 0.5
            else:
                # Normalizza i valori da 0.0 (peggiore) a 1.0 (migliore) dentro la singola colonna
                df_norm[col] = (df_numeric[col] - col_min) / (col_max - col_min)
                
            # Trova il massimo assoluto della colonna e forzalo a 2.0 (Valore di "rottura" per il bianco)
            max_idx = df_numeric[col].idxmax()
            if pd.notna(max_idx):
                df_norm.loc[max_idx, col] = 2.0

        df_norm = df_norm.fillna(0.5)

        # Creazione dei testi per le celle (grassetto per i massimi)
        text_array = []
        for r in df_matrix.index:
            row_text = []
            for c in df_matrix.columns:
                val = df_matrix.loc[r, c]
                t = f"{val:+.2f}%" if pd.notna(val) else "N/D"
                
                # Grassetto se è il vincitore della colonna
                if pd.notna(val) and val == df_numeric[c].max():
                    row_text.append(f"<b>{t}</b>")
                else:
                    row_text.append(t)
            text_array.append(row_text)

        # Rendering Plotly
        fig_hm = go.Figure(data=go.Heatmap(
            z=df_norm.values, 
            x=df_matrix.columns,
            y=df_matrix.index,
            colorscale=[
                [0.00, '#ef4444'],   # 0.0 mappa al peggiore (Rosso)
                [0.25, '#fef08a'],   # 0.5 mappa ai valori intermedi (Giallo)
                [0.50, '#22c55e'],   # 1.0 mappa ai migliori "non vincitori" (Verde)
                [0.5001, '#ffffff'], # Subito dopo l'1.0, rompe la scala
                [1.00, '#ffffff']    # 2.0 mappa al vincitore assoluto (Bianco candido)
            ],
            zmin=0.0,
            zmax=2.0, # Scala estesa da 0 a 2 per gestire il bianco
            text=text_array,
            texttemplate="%{text}",
            showscale=False,
            xgap=2, ygap=2
        ))
        
        fig_hm.update_layout(
            template='plotly_dark', 
            margin=dict(l=0, r=0, t=10, b=0),
            height=500, 
            xaxis=dict(side='top', tickfont=dict(size=12, color="#cbd5e1")),
            yaxis=dict(tickfont=dict(size=11, color="#f8fafc"), autorange="reversed"),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_hm, use_container_width=True)
    else:
        st.warning("⚠️ Dati insufficienti per renderizzare la matrice dei regimi.")

    st.divider()
     
    # ==========================================================
    # MODULO CICLO ECONOMICO E HARD VETO
    # ==========================================================
    st.markdown("### 🧭 Posizionamento nel Ciclo Economico")
    with st.spinner("Estrazione tassi di rendimento e computo delle pendenze..."):
        fase_attuale, raw_phase, veto_applied, macro_metrics = calculate_macro_cycle_phase(df, dominant_regime)

    if veto_applied:
        st.markdown(f"""
            <div style="background-color: #451A03; border-left: 5px solid #F59E0B; padding: 15px; border-radius: 4px; color: #FDE68A; margin-bottom: 20px;">
                ⚠️ <b>VETO ALGORITMICO APPLICATO:</b> L'algoritmo indicava originariamente '{raw_phase}'. L'output è stato forzato matematicamente a '{fase_attuale}' a causa del regime '{dominant_regime}'.
            </div>
        """, unsafe_allow_html=True)

    if fase_attuale != "DATI INSUFFICIENTI" and fase_attuale != "ERRORE DATI":
        quad_cols = st.columns(4)
        fasi_ciclo = ["Ripresa", "Espansione", "Picco / Stagflazione", "Contrazione"]
        for i, fase in enumerate(fasi_ciclo):
            with quad_cols[i]:
                is_active = (fase.lower() == fase_attuale.lower())
                bg_color = "#0f766e" if is_active else "transparent"
                border_color = "#14b8a6" if is_active else "#334155"
                text_color = "#ffffff" if is_active else "#64748b"
                
                st.markdown(
                    f"""
                    <div style="background-color: {bg_color}; padding: 16px; border-radius: 6px; text-align: center; border: 1px solid {border_color};">
                        <h4 style="color: {text_color}; margin: 0; font-size: 14px; font-weight: 700; text-transform: uppercase;">{fase}</h4>
                    </div>
                    """, unsafe_allow_html=True
                )
        st.write("")
        
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Spread 10Y-2Y", f"{macro_metrics.get('Spread_10Y_2Y', 0):.2f} pts")
        mc2.metric("Pendenza Rame/Oro (40D)", f"{macro_metrics.get('Pendenza_Cu_Au_40D', 0):+.3f}")
        
        ism_val = macro_metrics.get('ISM_PMI', 'N/D')
        ism_status = "🟢 ESPANSIONE" if isinstance(ism_val, float) and ism_val >= 50 else ("🔴 CONTRAZIONE" if isinstance(ism_val, float) else "⚪ NEUTRO")
        mc3.metric("US ISM PMI", f"{ism_val}", ism_status, delta_color="normal" if isinstance(ism_val, float) and ism_val >= 50 else "inverse")
        
        mc4.metric("Z-Score Tassi Reali", f"{macro_metrics.get('Z_Score_Tassi_Reali', 0):+.2f} σ")
        mc5.metric("Z-Score 30Y Treasury", f"{macro_metrics.get('Z_Score_30Y_Yield', 0):+.2f} σ")
    else:
        st.warning(f"⚠️ {fase_attuale}: Impossibile calcolare il ciclo. Assicurati di aver sincronizzato i flussi API istituzionali.")

    st.divider()

    # ==========================================================
    # GRAFICI MACRO STRUTTURALI
    # ==========================================================
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<h4 style='font-size:15px; color:#cbd5e1; font-weight: 600;'>1. Liquidità Netta Estesa (Miliardi $)</h4>", unsafe_allow_html=True)
        if 'Net_Liquidity' in df.columns and not df['Net_Liquidity'].dropna().empty:
            df_liq = df.dropna(subset=['Net_Liquidity']).tail(250).copy()
            df_liq['Net_Liquidity_Bil'] = df_liq['Net_Liquidity'] / 1000
            st.plotly_chart(px.area(df_liq, x="Data", y="Net_Liquidity_Bil", color_discrete_sequence=['#14b8a6'], template='plotly_dark'), use_container_width=True)
    with c2:
        st.markdown("<h4 style='font-size:15px; color:#cbd5e1; font-weight: 600;'>2. M2 Money Supply</h4>", unsafe_allow_html=True)
        if 'M2' in df.columns and not df['M2'].dropna().empty:
            st.plotly_chart(px.line(df.dropna(subset=['M2']).tail(250), x="Data", y="M2", template='plotly_dark'), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("<h4 style='font-size:15px; color:#cbd5e1; font-weight: 600;'>3. Modello GOLD / OIL</h4>", unsafe_allow_html=True)
        if 'Ratio_GO' in df.columns and not df['Ratio_GO'].dropna().empty:
            fig_go = px.line(df.dropna(subset=['Ratio_GO']).tail(100), x="Data", y="Ratio_GO", color_discrete_sequence=['#fbbf24'], template='plotly_dark')
            fig_go.add_hline(y=2.5, line_dash="dash", line_color="#ef4444")
            st.plotly_chart(fig_go, use_container_width=True)
    with c4:
        st.markdown("<h4 style='font-size:15px; color:#cbd5e1; font-weight: 600;'>4. Tassi vs Volatilità (TLT/MOVE)</h4>", unsafe_allow_html=True)
        if set(['TLT', 'MOVE']).issubset(df.columns):
            temp_df = df.dropna(subset=['TLT', 'MOVE']).tail(100)
            if not temp_df.empty:
                st.plotly_chart(px.line(temp_df, x="Data", y=["TLT", "MOVE"], color_discrete_map={"TLT": "#fbbf24", "MOVE": "#ef4444"}, template='plotly_dark'), use_container_width=True)

    st.divider()
    
    st.markdown("### Tabella Master EOD")
    display_df = df.sort_values("Data", ascending=False).head(30).copy()
    display_df['Data'] = display_df['Data'].dt.strftime('%Y-%m-%d')
    st.dataframe(display_df, use_container_width=True, hide_index=True)
