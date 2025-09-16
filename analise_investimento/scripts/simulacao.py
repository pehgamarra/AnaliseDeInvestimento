import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

plt.style.use('seaborn-v0_8-whitegrid')

capital_inicial = 100000
ANNUAL_DAYS = 252


# ======= Funções financeiras =======
def anualizar_retorno(retornos_d):
    return retornos_d.mean() * ANNUAL_DAYS

def anualizar_vol(retornos_d):
    return retornos_d.std() * np.sqrt(ANNUAL_DAYS)

def sharpe_ratio(retornos_d, rf=0.0):
    mu = retornos_d.mean() - rf
    sr = mu / retornos_d.std()
    return (sr * np.sqrt(ANNUAL_DAYS))

def sortino_ratio(retornos_d, rf=0.0):
    # usa downside deviation (diário) e annualiza
    downside = retornos_d[retornos_d < 0]
    if len(downside) == 0:
        return np.nan
    dr = downside.std()
    return ((retornos_d.mean() - rf) / dr) * np.sqrt(ANNUAL_DAYS)

def carteira_metricas(pesos, retornos):
    """Retorna (retorno_anual, vol_anual, sharpe) para um vetor de pesos"""
    pesos = np.array(pesos)
    mu_ann = retornos.mean().values @ pesos * ANNUAL_DAYS
    cov_ann = retornos.cov().values * ANNUAL_DAYS
    vol_ann = np.sqrt(pesos.T @ cov_ann @ pesos)
    sharpe = mu_ann / vol_ann if vol_ann != 0 else 0
    return mu_ann, vol_ann, sharpe

def drawdown_series(serie_valor_acumulado):
    roll_max = serie_valor_acumulado.cummax()
    dd = (serie_valor_acumulado - roll_max) / roll_max
    return dd

def max_drawdown(serie_valor_acumulado):
    return drawdown_series(serie_valor_acumulado).min()


# ======= Geração de carteiras aleatórias (exploratória) =======
def gerar_carteiras_aleatorias(n_portfs, retornos, ativos):
    n = len(ativos)
    results = []
    pesos_list = []
    for _ in range(n_portfs):
        w = np.random.random(n)
        w /= w.sum()
        mu, vol, sr = carteira_metricas(w, retornos)
        results.append({'Retorno': mu, 'Risco': vol, 'Sharpe': sr})
        pesos_list.append(w)
    df = pd.DataFrame(results)
    for i, a in enumerate(ativos):
        df[a] = [w[i] for w in pesos_list]
    return df

# ======= Otimizações: melhor Sharpe e maior risco =======
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
    # maximize volatility -> minimize negative volatility
    n = len(ativos)
    x0 = np.repeat(1/n, n)
    limites = tuple((bounds[0], bounds[1]) for _ in range(n))
    restr = ({'type':'eq', 'fun': lambda x: np.sum(x)-1})
    def negativo_vol(x):
        return -carteira_metricas(x, retornos)[1]
    res = minimize(negativo_vol, x0, method='SLSQP', bounds=limites, constraints=restr)
    return res.x


# ======= Simulação de valor (com/sem rebalanceamento) =======
def simular_valor_carteira(pesos, dados_precos, capital=capital_inicial, rebalance=False, freq='M'):
    """
    Retorna:
      - serie_valor_total (pd.Series com índice dos preços)
      - alocacao_inicial (array R$)
      - valores_finais_por_ativo (pd.Series)
      - holdings_df (DataFrame diário com valor por ativo)  (útil p/ rebalance)
    """
    ativos = dados_precos.columns
    # quantidade inicial de "unidades" compradas se investirmos capital * pesos no preço de abertura (primeiro dia)
    preco_inicial = dados_precos.iloc[0]
    invest_inicial = pesos * capital
    unidades = invest_inicial / preco_inicial.values  # quantity of each asset
    # daily values per asset
    valores_diarios = dados_precos * unidades  # DataFrame: cada coluna = valor desse ativo ao longo do tempo
    serie_valor_total = valores_diarios.sum(axis=1)

    if rebalance:
        # Rebalance: aqui implementamos rebalance para pesos alvo em freq (mensal/trimestral)
        df_holdings = pd.DataFrame(index=dados_precos.index, columns=ativos, dtype=float)
        current_unidades = unidades.copy()
        for date in dados_precos.index:
            # valor atual por ativo
            current_values = dados_precos.loc[date] * current_unidades
            total = current_values.sum()
            df_holdings.loc[date] = current_values
        unidades = unidades.copy()
        for period_end_idx in dados_precos.resample(freq).apply(lambda x: x.index[-1]).values:
            pass
        return serie_valor_total, invest_inicial, valores_diarios.iloc[-1], valores_diarios
    else:
        return serie_valor_total, invest_inicial, valores_diarios.iloc[-1], valores_diarios


# ======= Visualizações / Relatórios =======
def plot_evolucao_duas_carteiras(serie1, label1, serie2, label2, capital=capital_inicial):
    plt.figure(figsize=(12,6))
    plt.plot(serie1.index, serie1.values, label=label1, linewidth=2)
    plt.plot(serie2.index, serie2.values, label=label2, linewidth=2, linestyle='--')
    plt.axhline(capital, color='gray', linestyle=':', label='Capital Inicial')
    plt.title("Evolução do Capital - Comparação")
    plt.ylabel("Valor (R$)")
    plt.xlabel("Data")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_inicial_final_por_ativo(ax, ativos, inicial_vals, final_vals, pesos, color_init='#8EC6FF', color_final='#005b96', titulo=""):
    n = len(ativos)
    x = np.arange(n)
    largura = 0.35
    ax.bar(x - largura/2, inicial_vals, width=largura, color=color_init, edgecolor='black', label='Inicial (R$)')
    ax.bar(x + largura/2, final_vals, width=largura, color=color_final, edgecolor='black', label='Final (R$)')
    for i, (ini, fin, p) in enumerate(zip(inicial_vals, final_vals, pesos)):
        pct = (fin/ini - 1) * 100 if ini != 0 else np.nan
        ax.text(i + largura/2, fin + capital_inicial * 0.005, f"R$ {fin:,.0f}\n({pct:+.1f}%)", ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(ativos, rotation=45, ha='right')
    ax.set_title(titulo)
    ax.set_ylabel("Valor (R$)")
    ax.legend()

def plot_drawdown(serie_valor, titulo="Drawdown da Carteira"):
    cum = serie_valor / serie_valor.iloc[0]
    dd = drawdown_series(cum)
    plt.figure(figsize=(10,5))
    dd.plot(color='crimson')
    plt.title(titulo)
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.show()

def plot_vol_rolling(serie_retorno, janela=30, titulo=None):
    vol = serie_retorno.rolling(janela).std() * np.sqrt(ANNUAL_DAYS)
    plt.figure(figsize=(10,5))
    vol.plot()
    plt.title(titulo or f"Volatilidade Rolling ({janela} dias) - annualizada")
    plt.ylabel("Volatilidade anualizada")
    plt.tight_layout()
    plt.show()

def heatmap_retorno_mensal(serie_retorno, titulo):
    mensal = (1 + serie_retorno).resample('M').prod() - 1
    tabela = mensal.to_frame('ret').assign(
        year=mensal.index.year,
        month=mensal.index.month
    )
    tabela_pivot = tabela.pivot(index='year', columns='month', values='ret')
    plt.figure(figsize=(12,6))
    sns.heatmap(tabela_pivot, annot=True, fmt=".2%", cmap="RdYlGn", center=0)
    plt.title(titulo)
    plt.ylabel("Ano")
    plt.xlabel("Mês")
    plt.show()


def contribuicao_por_ativo(retornos, pesos):
    contrib = retornos.mean() * pesos * ANNUAL_DAYS
    contrib = contrib.sort_values(ascending=False)
    plt.figure(figsize=(10,5))
    contrib.plot(kind='bar')
    plt.title("Contribuição anualizada ao retorno por ativo")
    plt.ylabel("Contribuição anualizada (p. decimais)")
    plt.tight_layout()
    plt.show()
    return contrib


# ======= Pipeline: roda tudo para as duas carteiras =======
def relatorio_completo(dados, ativos, capital=capital_inicial, n_portfs=10000, rf_daily=0.0, benchmark_series=None):
    """
    Roda todo o fluxo e mostra gráficos para:
     - carteira otimizada por Sharpe
     - carteira otimizada por maior risco
    """
    retornos = dados[ativos].pct_change().dropna()

    # 1) gerar portf. aleatórias (útil para debug/visual)
    print("Gerando carteiras aleatórias...")
    df_rand = gerar_carteiras_aleatorias(n_portfs, retornos, ativos)

    # 2) otimizações
    print("Otimizando melhor Sharpe...")
    pesos_sharpe = otimizar_melhor_sharpe(retornos, ativos)
    mu_s, vol_s, sr_s = carteira_metricas(pesos_sharpe, retornos)

    print("Encontrando carteira de MAIOR risco (otimização)...")
    pesos_risco = otimizar_maior_risco(retornos, ativos)
    mu_r, vol_r, sr_r = carteira_metricas(pesos_risco, retornos)

    # 3) simular valores
    serie_s, ini_s, final_vals_s, daily_vals_s = simular_valor_carteira(pesos_sharpe, dados[ativos], capital, rebalance=False)
    serie_r, ini_r, final_vals_r, daily_vals_r = simular_valor_carteira(pesos_risco, dados[ativos], capital, rebalance=False)

    # 4) imprimir sumário
    print("\n=== RESUMO CARTEIRA MELHOR SHARPE ===")
    print(f"Retorno anual esperado: {mu_s:.2%} | Vol anual: {vol_s:.2%} | Sharpe: {sr_s:.3f}")
    for a, p, inv in zip(ativos, pesos_sharpe, ini_s):
        print(f" - {a}: {p*100:.2f}%  -> R$ {inv:,.2f}")

    print("\n=== RESUMO CARTEIRA MAIOR RISCO ===")
    print(f"Retorno anual esperado: {mu_r:.2%} | Vol anual: {vol_r:.2%} | Sharpe: {sr_r:.3f}")
    for a, p, inv in zip(ativos, pesos_risco, ini_r):
        print(f" - {a}: {p*100:.2f}%  -> R$ {inv:,.2f}")

    # 5) gráficos comparativos evolução
    plot_evolucao_duas_carteiras(serie_s, "Melhor Sharpe", serie_r, "Maior Risco", capital=capital)

    # 6) barras Inicial vs Final por ativo (duplo subplot)
    fig, axs = plt.subplots(1, 2, figsize=(16,6), sharey=True)
    plot_inicial_final_por_ativo(axs[0], ativos, ini_s, final_vals_s.values, pesos_sharpe, titulo="Melhor Sharpe")
    plot_inicial_final_por_ativo(axs[1], ativos, ini_r, final_vals_r.values, pesos_risco, titulo="Maior Risco")
    plt.tight_layout()
    plt.show()

    # 7) drawdown plots
    plot_drawdown(serie_s, "Drawdown - Melhor Sharpe")
    plot_drawdown(serie_r, "Drawdown - Maior Risco")

    # 8) volatilidade rolling (30 dias)
    plot_vol_rolling(retornos.dot(pesos_sharpe), janela=30, titulo="Volatilidade Rolling - Carteira Melhor Sharpe")
    plot_vol_rolling(retornos.dot(pesos_risco), janela=30, titulo="Volatilidade Rolling - Carteira Maior Risco")

    # 9) heatmap mensal de retornos
    heatmap_retorno_mensal(retornos.dot(pesos_sharpe), titulo="Retornos Mensais - Melhor Sharpe")
    heatmap_retorno_mensal(retornos.dot(pesos_risco), titulo="Retornos Mensais - Maior Risco")

    # 10) contribuicao por ativo
    print("\nContribuição (Melhor Sharpe):")
    contrib_s = contribuicao_por_ativo(retornos, pesos_sharpe)
    print("\nContribuição (Maior Risco):")
    contrib_r = contribuicao_por_ativo(retornos, pesos_risco)

    # 11) métricas finais (Sharpe/Sortino/Drawdown)
    print("\nMétricas finais:")
    print("Melhor Sharpe - Sharpe (ann):", sr_s)
    print("Melhor Sharpe - Sortino (ann):", sortino_ratio(retornos.dot(pesos_sharpe)))
    print("Melhor Sharpe - Max Drawdown:", max_drawdown(serie_s / serie_s.iloc[0]))

    print("\nMaior Risco - Sharpe (ann):", sr_r)
    print("Maior Risco - Sortino (ann):", sortino_ratio(retornos.dot(pesos_risco)))
    print("Maior Risco - Max Drawdown:", max_drawdown(serie_r / serie_r.iloc[0]))

    # 12) Comparação com benchmark (opcional)
    if benchmark_series is not None:
        # benchmark_series pode ser pd.Series alinhada com dados.index
        plt.figure(figsize=(10,6))
        (serie_s/serie_s.iloc[0]).plot(label='Melhor Sharpe (norm)')
        (serie_r/serie_r.iloc[0]).plot(label='Maior Risco (norm)')
        (benchmark_series/benchmark_series.iloc[0]).plot(label='Benchmark (norm)')
        plt.title("Comparação Normalizada com Benchmark")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Retorna objetos úteis
    out = {
        'pesos_sharpe': pesos_sharpe,
        'pesos_risco': pesos_risco,
        'serie_sharpe': serie_s,
        'serie_risco': serie_r,
        'inicial_sharpe': ini_s,
        'inicial_risco': ini_r,
        'final_vals_sharpe': final_vals_s,
        'final_vals_risco': final_vals_r,
        'df_random': df_rand
    }
    return out