import pandas as pd
from scipy.stats import skew, kurtosis


def carregar_dados(caminho_csv):
    """Carrega e prepara os dados do CSV"""
    dados = pd.read_csv(caminho_csv)
    dados['Date'] = pd.to_datetime(dados['Date'])
    dados.set_index('Date', inplace=True)
    return dados.dropna()


def calcular_drawdown(serie):
    """Calcula drawdown de uma série"""
    topo = serie.cummax()
    return (serie - topo) / topo


def metricas_risco_retorno(dados, ativos):
    """Calcula e exibe métricas de risco/retorno"""
    retornos = dados[ativos].pct_change().dropna()
    
    retorno_anual = retornos.mean() * 252
    volatilidade_anual = retornos.std() * (252 ** 0.5)
    sharpe = retorno_anual / volatilidade_anual
    
    df_metricas = pd.DataFrame({
        'Retorno Anual (%)': (retorno_anual * 100).round(2),
        'Volatilidade (%)': (volatilidade_anual * 100).round(2),
        'Sharpe': sharpe.round(2)
    })
    
    print(" MÉTRICAS DE RISCO E RETORNO:")
    print(df_metricas)
    
    return df_metricas


def valorizacao_total(dados, ativos):
    """Calcula valorização total no período"""
    import plotly.express as px
    
    valorizacoes = {}
    for ativo in ativos:
        inicio = dados[ativo].iloc[0]
        fim = dados[ativo].iloc[-1]
        retorno = ((fim - inicio) / inicio) * 100
        valorizacoes[ativo] = round(retorno, 2)
    
    print("VALORIZAÇÃO TOTAL:")
    for ativo, ret in sorted(valorizacoes.items(), key=lambda x: x[1], reverse=True):
        print(f"   {ativo}: {ret}%")
    
    # Gráfico interativo
    df_val = pd.DataFrame.from_dict(valorizacoes, orient='index', columns=['Retorno (%)'])
    df_val = df_val.sort_values('Retorno (%)')
    
    fig = px.bar(
        df_val, x='Retorno (%)', y=df_val.index, orientation='h',
        title='Valorização Total por Ativo', color='Retorno (%)',
        color_continuous_scale='RdYlGn'
    )
    fig.show()


def analise_drawdown(dados, ativos):
    """Análise de drawdown dos ativos"""
    import matplotlib.pyplot as plt
    
    precos_norm = dados[ativos] / dados[ativos].iloc[0]
    drawdowns = precos_norm.apply(calcular_drawdown)
    drawdown_max = (drawdowns.min() * 100).round(2)
    
    print("DRAWDOWN MÁXIMO:")
    for ativo, dd in drawdown_max.sort_values().items():
        print(f"   {ativo}: {dd}%")
    
    # Gráfico de barras
    plt.figure(figsize=(10, 6))
    drawdown_max.sort_values().plot(kind='barh', color='crimson')
    plt.title('Drawdown Máximo por Ativo')
    plt.xlabel('Drawdown (%)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return drawdowns


def relatorio_resumido(dados, ativos):
    """Gera relatório resumido"""
    print("="*50)
    print("RELATÓRIO DE ANÁLISE FINANCEIRA")
    print("="*50)
    print(f"Período: {dados.index[0].date()} até {dados.index[-1].date()}")
    print(f"Total de dias: {len(dados)}")
    print(f"Ativos analisados: {len(ativos)}")
    print()
    
    # Métricas principais
    df_metricas = metricas_risco_retorno(dados, ativos)
    print()
    
    # Drawdown
    analise_drawdown(dados, ativos)
    print()
    
    # Valorização
    valorizacao_total(dados, ativos)
    print()
    
    # Melhor e pior ativo
    retornos_total = {}
    for ativo in ativos:
        inicio = dados[ativo].iloc[0]
        fim = dados[ativo].iloc[-1]
        retornos_total[ativo] = ((fim - inicio) / inicio) * 100
    
    melhor = max(retornos_total, key=retornos_total.get)
    pior = min(retornos_total, key=retornos_total.get)
    
    print("DESTAQUES:")
    print(f"   Melhor performance: {melhor} ({retornos_total[melhor]:.2f}%)")
    print(f"   Pior performance: {pior} ({retornos_total[pior]:.2f}%)")
    print("="*50)