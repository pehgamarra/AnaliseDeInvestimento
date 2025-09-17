import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

sns.set_style("whitegrid")

capital_inicial = 100000
ANNUAL_DAYS = 252

def carteira_metricas(pesos, retornos):
    pesos = np.array(pesos)
    mu_ann = retornos.mean().values @ pesos * ANNUAL_DAYS
    cov_ann = retornos.cov().values * ANNUAL_DAYS
    vol_ann = np.sqrt(pesos.T @ cov_ann @ pesos)
    sharpe = mu_ann / vol_ann if vol_ann != 0 else 0
    return mu_ann, vol_ann, sharpe

def gerar_carteiras_aleatorias(n_portfs, retornos, ativos):
    n = len(ativos)
    results = []
    pesos_list = []
    for _ in range(n_portfs):
        w = np.random.random(n)
        w = w / w.sum()
        mu, vol, sr = carteira_metricas(w, retornos)
        results.append({'Retorno': mu, 'Risco': vol, 'Sharpe': sr})
        pesos_list.append(w)
    df = pd.DataFrame(results)
    for i, a in enumerate(ativos):
        df[a] = [w[i] for w in pesos_list]
    return df

def otimizar_melhor_sharpe(retornos, ativos, bounds=(0,1)):
    n = len(ativos)
    x0 = np.repeat(1/n, n)
    limites = tuple((bounds[0], bounds[1]) for _ in range(n))
    restr = ({'type':'eq', 'fun': lambda x: np.sum(x)-1})
    def negativo_sharpe(x):
        return -carteira_metricas(x, retornos)[2]
    res = minimize(negativo_sharpe, x0, method='SLSQP', bounds=limites, constraints=restr)
    return res.x

def otimizar_maior_risco(retornos, ativos, bounds=(0,1)):
    n = len(ativos)
    x0 = np.repeat(1/n, n)
    limites = tuple((bounds[0], bounds[1]) for _ in range(n))
    restr = ({'type':'eq', 'fun': lambda x: np.sum(x)-1})
    def negativo_vol(x):
        return -carteira_metricas(x, retornos)[1]
    res = minimize(negativo_vol, x0, method='SLSQP', bounds=limites, constraints=restr)
    return res.x

def simular_valor_carteira(pesos, dados_precos, capital=capital_inicial, rebalance=False, freq='M'):
    """
    Retorna:
      - serie_valor_total (pd.Series)
      - alocacao_inicial (np.array em R$)
      - valores_finais_por_ativo (pd.Series em R$)
      - valores_diarios (DataFrame valor por ativo)
    Implementa rebalanceamento simples no final de cada período (freq), ajustando unidades.
    """
    ativos = list(dados_precos.columns)
    preco_inicial = dados_precos.iloc[0]
    pesos = np.array(pesos)
    invest_inicial = pesos * capital
    unidades = invest_inicial / preco_inicial.values

    # cria DataFrame com valores diários por ativo
    valores_diarios = dados_precos.multiply(unidades, axis=1)
    serie_valor_total = valores_diarios.sum(axis=1)

    if not rebalance:
        alocacao_inicial = invest_inicial
        valores_finais = valores_diarios.iloc[-1]
        return serie_valor_total, alocacao_inicial, valores_finais, valores_diarios
    else:
        # Rebalance mensal (ou outra frequência): no último dia de cada period, recalcula unidades para voltar aos pesos alvo
        df_holdings = pd.DataFrame(index=dados_precos.index, columns=ativos, dtype=float)
        unidades = unidades.copy()
        current_unidades = unidades.copy()
        grouped = list(dados_precos.resample(freq))
        last_idxs = [g[1].index[-1] for g in grouped if len(g[1])>0]

        for date in dados_precos.index:
            values = dados_precos.loc[date] * current_unidades
            df_holdings.loc[date] = values
            # se é último dia do período, rebalanceia: converte total para target pesos e calcula novas unidades
            if date in last_idxs:
                total = values.sum()
                target_value = total * pesos  # target in R$
                # avoid division by zero
                new_unidades = np.where(dados_precos.loc[date].values != 0, target_value / dados_precos.loc[date].values, current_unidades)
                current_unidades = new_unidades

        serie_valor_total = df_holdings.sum(axis=1)
        alocacao_inicial = invest_inicial
        valores_finais = df_holdings.iloc[-1]
        return serie_valor_total, alocacao_inicial, valores_finais, df_holdings

# ---- visual helpers ----
def fig_evolucao(serie, titulo="Evolução do Capital", capital_inicial=capital_inicial):
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(serie.index, serie.values, linewidth=2)
    ax.axhline(capital_inicial, color='gray', linestyle='--', label='Capital Inicial')
    ax.set_title(titulo)
    ax.set_ylabel("Valor (R$)")
    ax.set_xlabel("Data")
    ax.legend()
    fig.tight_layout()
    return fig

def fig_inicial_final_por_ativo(ativos, inicial_vals, final_vals, pesos, titulo="Inicial vs Final por Ativo"):
    fig, ax = plt.subplots(figsize=(10,5))
    x = np.arange(len(ativos))
    largura = 0.35
    ax.bar(x - largura/2, inicial_vals, width=largura, label='Inicial', color='#8EC6FF', edgecolor='black')
    ax.bar(x + largura/2, final_vals, width=largura, label='Final', color='#005b96', edgecolor='black')
    for i, (ini, fin, p) in enumerate(zip(inicial_vals, final_vals, pesos)):
        pct = (fin/ini - 1)*100 if ini != 0 else np.nan
        ax.text(i - largura/2, ini + 0.005 * np.max(inicial_vals), 
                f"R$ {ini:,.0f}", ha='center', va='bottom', fontsize=6)
        ax.text(i + largura/2, fin + 0.005 * np.max(final_vals), 
                f"R$ {fin:,.0f}\n({pct:+.1f}%)", ha='center', va='bottom', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(ativos, rotation=45, ha='right')
    ax.set_title(titulo)
    ax.set_ylabel("Valor (R$)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_vol_rolling(serie_retorno, janela=30):
    vol = serie_retorno.rolling(janela).std() * np.sqrt(ANNUAL_DAYS)
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(vol.index, vol.values)
    ax.set_title(f"Volatilidade Rolling ({janela} dias) - anualizada")
    ax.set_ylabel("Volatilidade")
    fig.tight_layout()
    return fig

def heatmap_retorno_mensal(serie_retorno, titulo="Retornos Mensais"):
    mensal = (1 + serie_retorno).resample('M').prod() - 1
    tabela = mensal.to_frame('ret').assign(year=mensal.index.year, month=mensal.index.month)
    tabela_pivot = tabela.pivot(index='year', columns='month', values='ret')
    fig, ax = plt.subplots(figsize=(12,6))
    sns.heatmap(tabela_pivot, annot=True, fmt=".2%", cmap="RdYlGn", center=0, ax=ax)
    ax.set_title(titulo)
    fig.tight_layout()
    return fig

def contribuicao_por_ativo(retornos, pesos):
    contrib = retornos.mean() * pesos * ANNUAL_DAYS
    contrib = contrib.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10,4))
    contrib.plot(kind='bar', ax=ax)
    ax.set_title("Contribuição anualizada ao retorno por ativo")
    ax.set_ylabel("Contribuição anualizada")
    fig.tight_layout()
    return fig, contrib
