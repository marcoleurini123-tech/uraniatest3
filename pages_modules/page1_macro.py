import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from backend_engine import (
    load_db, save_db, fetch_yahoo_data, fetch_bridge_data, 
    fetch_squeezemetrics_data, fetch_cboe_pc_ratio, COLUMNS,
    fetch_regime_baskets_data, calculate_regime_matrix,
    fetch_macro_cycle_data, calculate_macro_cycle_phase,
    calculate_risk_propensity
)

@st.cache_data(ttl=3600)
def get_cached_regime_data():
    return fetch_regime_baskets_data(period="2y")

@st.cache_data(ttl=3600)
def get_cached_macro_data():
    return fetch_macro_cycle_data()

def render_page1():
    st.title("🛡️ Terminale Macro Professionale")
    st.caption("Motore Vettoriale Blindato • Analisi Divergenze Istituzionali")
    
    df = load_db()

    with st.sidebar:
        st.header("⚙️ Override Dati EOD")
        st.caption("I dati immessi manualmente non verranno sovrascritti dal fetch API.")
        
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
                st.success(f"Sessione {dt.strftime('%Y-%m-%d')} registrata.")
                st.rerun()

        st.divider()
        st.header("🔄 Fetch Istituzionale")
        
        if st.button("2. SINCRONIZZA FLUSSI API", use_container_width=True):
            with st.spinner("Allineamento matrici temporali in corso..."):
                d_y = fetch_yahoo_data(365)
                d_b = fetch_bridge_data()
                d_sq = fetch_squeezemetrics_data()
                d_pc = fetch_cboe_pc_ratio()
                
                fetched_df = pd.merge(d_y, d_b, on='Data', how='outer')
                if not d_sq.empty:
                    fetched_df = pd.merge(fetched_df, d_sq, on='Data', how='outer')
                if not d_pc.empty:
                    fetched_df = pd.merge(fetched_df, d_pc, on='Data', how='outer')

                if not df.empty:
                    fetched_df = fetched_df.set_index('Data')
                    local_df = df.set_index('Data')
                    final_df = local_df.combine_first(fetched_df).reset_index()
                else:
                    final_df = fetched_df

                for col in COLUMNS:
                    if col not in final_df.columns: final_df[col] = np.nan
                final_df = final_df[COLUMNS]

                final_df = final_df.sort_values("Data").ffill(limit=7).dropna(subset=['Data'])
                save_db(final_df)
                
                st.cache_data.clear()
                st.success("Sincronizzazione completata con successo.")
                st.rerun()

    if df.empty:
        st.warning("⚠️ Database locale vuoto. Esegui la sincronizzazione API.")
        return

    df = df.sort_values("Data").reset_index(drop=True)
    num_cols = [c for c in COLUMNS if c != "Data" and c in df.columns]
    df[num_cols] = df[num_cols].ffill(limit=7)

    df['Liq_Delta_5D'] = df['Net_Liquidity'].pct_change(periods=5) * 100
    df['Ratio_GO'] = np.where(df['USO'] > 0, df['GLD'] / df['USO'], np.nan)
    df['Ratio_Risk'] = np.where(df['XLP'] > 0, df['XLY'] / df['XLP'], np.nan)
    df['Ratio_Br'] = np.where(df['RSP'] > 0, df['SPY'] / df['RSP'], np.nan)

    last = df.iloc[-1]

    if len(df) >= 5 and last.get('Liq_Delta_5D', 0) < 0 and last.get('SPY', 0) > df.iloc[-5].get('SPY', 0):
        st.error(f"🚨 ALERT DIVERGENZA: Liquidità in calo ({last['Liq_Delta_5D']:.2f}%) mentre lo SPY sale. Pericolo di storno sistemico.")

    st.subheader("🚦 Monitor Segnali di Regime")
    r1, r2 = st.columns(6), st.columns(6)
    
    dix_v = last.get('DIX', np.nan)
    r1[0].metric("DIX", f"{dix_v:.1f}%" if not pd.isna(dix_v) else "N/A", "🟢 BULLISH" if dix_v > 45 else "⚪ NEUTRO")
    
    gex_v = last.get('GEX', np.nan)
    r1[1].metric("GEX", f"{gex_v:,.0f}" if not pd.isna(gex_v) else "N/A", "🔴 SQUEEZE" if gex_v < 0 else "🟢 STABILE", delta_color="inverse")
    
    pc_v = last.get('P_C', np.nan)
    pc_status = "🟢 PANICO" if pc_v > 1.05 else ("🔴 AVIDITÀ" if 0 < pc_v < 0.7 else "⚪ NEUTRO")
    r1[2].metric("P/C RATIO", f"{pc_v:.2f}" if not pd.isna(pc_v) else "N/A", pc_status)
    
    skew_v = last.get('SKEW', np.nan)
    r1[3].metric("SKEW", f"{skew_v:.1f}" if not pd.isna(skew_v) else "N/A", "⚠️ BLACK SWAN" if skew_v > 145 else "🟢 OK", delta_color="inverse")
    
    move_v = last.get('MOVE', np.nan)
    r1[4].metric("MOVE", f"{move_v:.1f}" if not pd.isna(move_v) else "N/A", "🔴 STRESS BOND" if move_v > 115 else "🟢 CALMO", delta_color="inverse")
    
    liq_d = last.get('Liq_Delta_5D', np.nan)
    liq_col = "normal" if not pd.isna(liq_d) and liq_d >= 0 else "inverse"
    r1[5].metric("Δ LIQ. 5D", f"{liq_d:.2f}%" if not pd.isna(liq_d) else "N/A", "📉 CONTRAZIONE" if not pd.isna(liq_d) and liq_d < 0 else "📈 ESPANSIONE", delta_color=liq_col)

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
    # PANORAMICA MACRO E MERCATI (STILE QUANTASTE)
    # ==========================================================
    with st.spinner("Estrazione dati macro e calcolo probabilità statistiche..."):
        df_regime_prices = get_cached_regime_data()
        df_matrix, dominant_regime, conf_pct = calculate_regime_matrix(df_regime_prices)
        risk_metrics, risk_err = calculate_risk_propensity(df)

    col_q1, col_q2 = st.columns(2)
    
    with col_q1:
        st.markdown(f"""
        <div style="background-color:#0f172a; padding:20px; border-radius:8px; border: 1px solid #334155;">
            <h4 style="color:#94a3b8; margin-top:0; font-size:14px; text-transform:uppercase;">Regime Economico Predominante</h4>
            <div style="display:flex; justify-content:space-between; align-items:flex-end;">
                <div>
                    <span style="color:#64748b; font-size:12px;">REGIME DOMINANTE</span><br>
                    <span style="color:#f8fafc; font-size:24px; font-weight:bold;">{dominant_regime}</span>
                </div>
                <div style="text-align:right;">
                    <span style="color:#f59e0b; font-size:32px; font-weight:bold;">{conf_pct}%</span>
                </div>
            </div>
            <div style="margin-top:15px; height:8px; width:100%; background: linear-gradient(90deg, #ef4444 0%, #f59e0b 50%, #10b981 100%); border-radius:4px;"></div>
        </div>
        """, unsafe_allow_html=True)

    with col_q2:
        if risk_err:
            r_status, r_on, r_off = "N/D", 0, 0
        else:
            r_status = risk_metrics['Status']
            r_on = risk_metrics['Risk_On_Pct']
            r_off = risk_metrics['Risk_Off_Pct']
            
        r_color = "#10b981" if r_status == "RISK ON" else ("#ef4444" if r_status == "RISK OFF" else "#f59e0b")
        
        st.markdown(f"""
        <div style="background-color:#0f172a; padding:20px; border-radius:8px; border: 1px solid #334155;">
            <h4 style="color:#94a3b8; margin-top:0; font-size:14px; text-transform:uppercase;">Propensione al Rischio</h4>
            <div style="display:flex; justify-content:space-between; align-items:flex-end;">
                <div>
                    <span style="color:#64748b; font-size:12px;">RISK ON / RISK OFF</span><br>
                    <span style="color:{r_color}; font-size:24px; font-weight:bold;">{r_status}</span>
                </div>
                <div style="text-align:right;">
                    <span style="color:#f59e0b; font-size:32px; font-weight:bold;">{r_on}%</span>
                </div>
            </div>
            <div style="margin-top:15px; display:flex; border-radius:4px; overflow:hidden; height:8px;">
                <div style="width:{r_off}%; background-color:#ef4444;"></div>
                <div style="width:{r_on}%; background-color:#10b981;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ==========================================================
    # MODULO A: MATRICE DEI 9 REGIMI MACRO
    # ==========================================================
    st.subheader("🗺️ Matrice dei Regimi di Mercato (Heatmap Z-Score)")

    if not df_matrix.empty:
        fig_hm = go.Figure(data=go.Heatmap(
            z=df_matrix.values,
            x=df_matrix.columns,
            y=df_matrix.index,
            colorscale='RdYlGn',
            text=df_matrix.map(lambda x: f"{x:.2f}%" if not pd.isna(x) else "N/D").values,
            texttemplate="%{text}",
            showscale=False
        ))
        fig_hm.update_layout(
            template='plotly_dark', 
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Orizzonte Temporale",
            yaxis_title="Regimi"
        )
        st.plotly_chart(fig_hm, use_container_width=True)
    else:
        st.warning("⚠️ Dati insufficienti per il calcolo della Matrice dei Regimi.")

    st.divider()

    # ==========================================================
    # MODULO B: QUADRANTI DEL CICLO ECONOMICO
    # ==========================================================
    st.subheader("🧭 Posizionamento nel Ciclo Economico")
    with st.spinner("Estrazione dati Federal Reserve e calcolo incroci macro..."):
        df_macro = get_cached_macro_data()
        fase_attuale, macro_metrics = calculate_macro_cycle_phase(df_macro)

    if fase_attuale != "DATI INSUFFICIENTI":
        quad_cols = st.columns(4)
        fasi_ciclo = ["RIPRESA", "ESPANSIONE", "PICCO / STAGFLAZIONE", "CONTRAZIONE"]

        for i, fase in enumerate(fasi_ciclo):
            with quad_cols[i]:
                bg_color = "#00CC96" if fase == fase_attuale else "transparent"
                border_color = "#00CC96" if fase == fase_attuale else "#334155"
                text_color = "#ffffff" if fase == fase_attuale else "#64748b"
                
                st.markdown(
                    f"""
                    <div style="background-color: {bg_color}; padding: 15px; border-radius: 6px; text-align: center; border: 1px solid {border_color};">
                        <h4 style="color: {text_color}; margin: 0; font-size: 16px;">{fase}</h4>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Spread 10Y-2Y", f"{macro_metrics.get('Spread_10Y_2Y', 0):.2f}%")
        mc2.metric("Rame/Oro Trend", macro_metrics.get('Copper_Gold_Trend', 'N/D'))
        mc3.metric("Tassi Reali (Z-Score)", f"{macro_metrics.get('Real_Rates_Z', 0):.2f}")
        
        debasement_status = "⚠️ ATTIVO" if macro_metrics.get('Debasement_Risk') else "🟢 INATTIVO"
        mc4.metric("Rischio Debasement 30Y", debasement_status)
    else:
        st.warning("⚠️ Dati FRED o Commodities insufficienti per inquadrare il ciclo economico.")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("💹 1. Vera Liquidità Netta (Trend)")
        if 'Net_Liquidity' in df.columns and not df['Net_Liquidity'].dropna().empty:
            st.plotly_chart(px.area(df.dropna(subset=['Net_Liquidity']).tail(250), x="Data", y="Net_Liquidity", color_discrete_sequence=['#00CC96'], template='plotly_dark'), use_container_width=True)
    
    with c2:
        st.subheader("💰 2. M2 Money Supply")
        if 'M2' in df.columns and not df['M2'].dropna().empty:
            st.plotly_chart(px.line(df.dropna(subset=['M2']).tail(250), x="Data", y="M2", template='plotly_dark'), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("🏆 3. Ratio GOLD / OIL")
        if 'Ratio_GO' in df.columns and not df['Ratio_GO'].dropna().empty:
            fig_go = px.line(df.dropna(subset=['Ratio_GO']).tail(100), x="Data", y="Ratio_GO", color_discrete_sequence=['#FFD700'], template='plotly_dark')
            fig_go.add_hline(y=2.5, line_dash="dash", line_color="red")
            st.plotly_chart(fig_go, use_container_width=True)
            
    with c4:
        st.subheader("📉 4. Bond: TLT Price vs MOVE")
        if set(['TLT', 'MOVE']).issubset(df.columns):
            temp_df = df.dropna(subset=['TLT', 'MOVE']).tail(100)
            if not temp_df.empty:
                st.plotly_chart(px.line(temp_df, x="Data", y=["TLT", "MOVE"], color_discrete_map={"TLT": "yellow", "MOVE": "red"}, template='plotly_dark'), use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        st.subheader("⚖️ 5. Ratio XLY / XLP")
        if 'Ratio_Risk' in df.columns and not df['Ratio_Risk'].dropna().empty:
            st.plotly_chart(px.line(df.dropna(subset=['Ratio_Risk']).tail(100), x="Data", y="Ratio_Risk", color_discrete_sequence=['#00D1FF'], template='plotly_dark'), use_container_width=True)
            
    with c6:
        st.subheader("⚖️ 6. Ratio SPY / RSP")
        if 'Ratio_Br' in df.columns and not df['Ratio_Br'].dropna().empty:
            fig_br = px.line(df.dropna(subset=['Ratio_Br']).tail(100), x="Data", y="Ratio_Br", color_discrete_sequence=['orange'], template='plotly_dark')
            fig_br.add_hline(y=3.5, line_dash="dash", line_color="red")
            st.plotly_chart(fig_br, use_container_width=True)

    c7, c8 = st.columns(2)
    with c7:
        st.subheader("📈 7. VIX Term Structure")
        vix_cols = ["VIX1D", "VIX9D", "VIX", "VIX3M", "VIX6M", "VIX1Y"]
        if set(vix_cols).issubset(df.columns):
            t_vals = [last.get(c, np.nan) for c in vix_cols]
            if not all(pd.isna(t_vals)):
                fig_vx = go.Figure(go.Scatter(
                    x=["1D","9D","30D","3M","6M","1Y"], 
                    y=t_vals, 
                    mode='lines+markers+text', 
                    text=[f"{v:.1f}" if not pd.isna(v) else "" for v in t_vals], 
                    textposition="top center",
                    connectgaps=True
                ))
                fig_vx.update_traces(line=dict(color="red" if last.get('VIX1D',0) > last.get('VIX',0) else "green", width=4))
                fig_vx.update_layout(template='plotly_dark', margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_vx, use_container_width=True)
                
    with c8:
        st.subheader("⚡ 8. VVIX vs DXY")
        if set(['VVIX', 'DXY']).issubset(df.columns):
            temp_df = df.dropna(subset=['VVIX', 'DXY']).tail(100)
            if not temp_df.empty:
                st.plotly_chart(px.line(temp_df, x="Data", y=["VVIX", "DXY"], color_discrete_map={"VVIX": "orange", "DXY": "white"}, template='plotly_dark'), use_container_width=True)

    st.divider()
    st.subheader("Tabella Master EOD & Serie Storiche Normalizzate")
    display_df = df.sort_values("Data", ascending=False).head(30).copy()
    display_df['Data'] = display_df['Data'].dt.strftime('%Y-%m-%d')
    st.dataframe(display_df, use_container_width=True, hide_index=True)
