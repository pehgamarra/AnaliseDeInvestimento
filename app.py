import sys
sys.path.append('analise_investimento/scripts')
sys.path.append('analise_investimento/scripts/streamlit')

import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

from coleta import ColetorDados
from analise import metricas_risco_retorno
from visualizacao_streamlit import (
    plotar_interativo, matriz_correlacao,
    grafico_risco_retorno, analise_drawdown, valorizacao_total, estatisticas_descritivas
)
from simulacao_streamlit import (
    gerar_carteiras_aleatorias,
    otimizar_melhor_sharpe,
    otimizar_maior_risco,
    simular_valor_carteira,
    carteira_metricas,
    fig_evolucao,
    fig_inicial_final_por_ativo,
    plot_vol_rolling,
    heatmap_retorno_mensal,
)

st.markdown(
    """
    <style>
    /* Aumentar a fonte das tabs */
    button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] p {
        font-size: 20px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.figsize": (8, 4),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.edgecolor": "#94A3B8",
    "axes.titleweight": "bold",
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "lines.linewidth": 2,
    "font.family": "sans-serif"
})


st.set_page_config(page_title="Análise & Simulação — Streamlit", layout="wide")
st.title("📈 Análise & Simulação de Carteiras")

# -------- Sidebar / inputs --------
st.sidebar.header("Configurações")
# Lê tickers do arquivo
acoes_default, cryptos_default = ColetorDados.carregar_tickers_txt("tickers.txt")

# Widgets
acoes_sel = st.sidebar.multiselect("Ações / ETFs", options=acoes_default, default=acoes_default[:3])
cryptos_sel = st.sidebar.multiselect("Criptomoedas", options=cryptos_default, default=cryptos_default[:2])

data_inicio = st.sidebar.date_input("Data início", value=dt.date(2020,1,1))
data_fim = st.sidebar.date_input("Data fim", value=dt.date.today())
capital = st.sidebar.number_input("Capital inicial (R$)", value=100000, step=1000)
n_portfs = st.sidebar.slider("Carteiras aleatórias", 1000, 10000, 3000, step=500)
rebalance = st.sidebar.checkbox("Rebalanceamento mensal", value=False)

# -----------------------------------------------------------
if st.sidebar.button("Rodar Análise"):

    if (not acoes_sel) and (not cryptos_sel):
        st.warning("Selecione pelo menos um ativo.")    
        st.stop()

    # ---------- COLETA ----------
    coletor = ColetorDados()
    coletor.limpar()
    if acoes_sel: coletor.adicionar_ativos(acoes_sel)
    if cryptos_sel: coletor.adicionar_criptos(cryptos_sel)

    try:
        dados = coletor.coletar_todos(start=data_inicio.strftime("%Y-%m-%d"),
                                     end=data_fim.strftime("%Y-%m-%d"))
    except Exception as e:
        st.error(f"Erro na coleta: {e}")
        st.stop()

    if dados.empty:
        st.error("Nenhum dado foi retornado.")
        st.stop()

    # ---------- MATCHING ----------
    available_cols = list(dados.columns)
    crypto_map = {'bitcoin':'BTC-USD','ethereum':'ETH-USD','cardano':'ADA-USD','solana':'SOL-USD'}
    ativos = []

    for a in acoes_sel:
        if a in available_cols: ativos.append(a)
    for c in cryptos_sel:
        candidates = [c, c.lower(), c.upper(), crypto_map.get(c.lower())]
        for cand in candidates:
            if cand and cand in available_cols:
                ativos.append(cand); break
        else:
            substr_matches = [col for col in available_cols if c.lower() in col.lower()]
            if substr_matches: ativos.append(substr_matches[0])

    ativos = list(dict.fromkeys(ativos)) or available_cols.copy()
    # ---------- PREPARAÇÃO ----------
    plot_data = dados[ativos].apply(pd.to_numeric, errors='coerce')
    plot_data = plot_data.fillna(method='ffill').fillna(method='bfill')
    valid_ativos = list(plot_data.columns)
    retornos = plot_data[valid_ativos].pct_change().dropna()

    
    # ---------- VISUALIZAÇÃO DOS ATIVOS ----------
    st.subheader("Ativos Selecionados")

    cols = st.columns(3)

    for i, ativo in enumerate(ativos):
        serie = plot_data[ativo].dropna()
        preco_atual = serie.iloc[-1]
        preco_ant = serie.iloc[-8] if len(serie) > 7 else serie.iloc[0]
        variacao = ((preco_atual / preco_ant) - 1) * 100 if preco_ant != 0 else 0

        cor = "green" if variacao >= 0 else "red"
        icone = "▲" if variacao >= 0 else "▼"

        with cols[i % 3]:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
                            padding: 18px; border-radius: 15px;
                            box-shadow: 0px 4px 12px rgba(0,0,0,0.3);
                            text-align: center; color: white;">
                    <h3 style="margin-bottom: 0px;padding-left: 30px;">{ativo}</h3>
                    <p style="font-size: 22px; font-weight: bold; color:#8EC6FF; margin:2px;">
                        R$ {preco_atual:,.2f}
                    </p>
                    <p style="font-size: 14px; margin:2px 0; color:{cor}; font-weight:600;">
                        {icone} {variacao:+.2f}%
                    </p>
                    <p style="font-size: 12px; opacity: 0.7; margin:30;">
                        7 dias atrás: R$ {preco_ant:,.2f}
                    </p>
                </div>
                <br>

                """,
                unsafe_allow_html=True
            )


    # ---------- ABAS PRINCIPAIS ----------
    tab1, tab2, tab3, tab4 = st.tabs(["Exploração", "Risco & Retorno", "Simulação", "Conclusões"])

    # --- TAB 1: Exploração ---
    with tab1:
        st.subheader("Performance Normalizada")
        fig_norm = plotar_interativo(plot_data, valid_ativos, titulo="Performance Normalizada")
        st.plotly_chart(fig_norm, use_container_width=True)


        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Correlação")
            fig_corr = matriz_correlacao(plot_data, valid_ativos)
            st.pyplot(fig_corr); plt.close(fig_corr)
        with col2:
            with st.expander("Estatísticas descritivas"):
                stats_df, fig_stats = estatisticas_descritivas(plot_data, valid_ativos)
                st.dataframe(stats_df)
                st.pyplot(fig_stats); plt.close(fig_stats)

    # --- TAB 2: Risco & Retorno ---
    with tab2:
        st.subheader("Métricas de Risco e Retorno")
        df_metricas = metricas_risco_retorno(plot_data, valid_ativos)
        st.dataframe(df_metricas)

        try:
            fig_rr = grafico_risco_retorno(df_metricas)
            st.plotly_chart(fig_rr, use_container_width=True)
        except: st.info("Não foi possível plotar scatter.")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Valorização Total")
            fig_val, df_val = valorizacao_total(plot_data, valid_ativos)
            st.plotly_chart(fig_val, use_container_width=True)
        with col2:
            st.subheader("Drawdown Máximo")
            fig_dd, _ = analise_drawdown(plot_data, valid_ativos)
            st.pyplot(fig_dd); plt.close(fig_dd)

    # --- TAB 3: Simulação ---
    with tab3:
        st.subheader("Carteiras Aleatórias")
        df_rand = gerar_carteiras_aleatorias(n_portfs, retornos, valid_ativos)
        st.dataframe(df_rand.head(10))

        pesos_sharpe = otimizar_melhor_sharpe(retornos, valid_ativos)
        pesos_risco = otimizar_maior_risco(retornos, valid_ativos)
        mu_s, vol_s, sr_s = carteira_metricas(pesos_sharpe, retornos)
        mu_r, vol_r, sr_r = carteira_metricas(pesos_risco, retornos)

        col1, col2, col3 = st.columns(3)
        col1.metric("Sharpe (ótima)", f"{sr_s:.2f}")
        col2.metric("Retorno (ótima)", f"{mu_s:.2%}")
        col3.metric("Volatilidade (ótima)", f"{vol_s:.2%}")

        st.subheader("Evolução do Capital")
        serie_s, ini_s, final_s, _ = simular_valor_carteira(pesos_sharpe, plot_data[valid_ativos], capital, rebalance)
        serie_r, ini_r, final_r, _ = simular_valor_carteira(pesos_risco, plot_data[valid_ativos], capital, rebalance)

        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(fig_evolucao(serie_s, "Melhor Sharpe", capital)); plt.close()
        with col2:
            st.pyplot(fig_evolucao(serie_r, "Maior Risco", capital)); plt.close()

        with st.expander("Mais análises"):
            st.pyplot(fig_inicial_final_por_ativo(valid_ativos, ini_s, final_s.values, pesos_sharpe, "Melhor Sharpe")); plt.close()
            st.pyplot(fig_inicial_final_por_ativo(valid_ativos, ini_r, final_r.values, pesos_risco, "Maior Risco")); plt.close()
            st.pyplot(plot_vol_rolling(retornos.dot(pesos_sharpe), janela=30)); plt.close()
            st.pyplot(heatmap_retorno_mensal(retornos.dot(pesos_sharpe), "Mensal - Sharpe")); plt.close()

    # --- TAB 4: Conclusões ---
    with tab4:
        st.markdown(f"""
        **Resumo**:
        - Melhor Sharpe → Retorno {mu_s:.2%}, Vol {vol_s:.2%}, Sharpe {sr_s:.2f}
        - Maior Risco → Retorno {mu_r:.2%}, Vol {vol_r:.2%}, Sharpe {sr_r:.2f}
        """)
        st.markdown("**Insights:**")
        st.markdown("- Carteira ótima tende a suavizar risco mantendo bom retorno.")
        st.markdown("- Rebalanceamento pode reduzir o drift dos pesos.")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:gray;'>© 2025 Análise de Investimentos by Pedro Gamarra !</p>", unsafe_allow_html=True)
