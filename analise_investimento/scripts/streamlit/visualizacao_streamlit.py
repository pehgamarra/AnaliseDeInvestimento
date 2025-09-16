import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

sns.set_style("whitegrid")

def calcular_drawdown(serie):
    topo = serie.cummax()
    return (serie - topo) / topo

def plotar_performance_normalizada(dados, ativos, titulo="Performance Normalizada (base 1.0)"):
    precos_norm = dados[ativos] / dados[ativos].iloc[0]
    fig, ax = plt.subplots(figsize=(14,6))
    for ativo in ativos:
        ax.plot(precos_norm.index, precos_norm[ativo], label=ativo, linewidth=1.7)
    ax.set_title(titulo)
    ax.set_xlabel("Data")
    ax.set_ylabel("Valor Normalizado")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig

def plotar_interativo(dados, ativos, titulo="Performance Interativa"):
    precos_norm = dados[ativos] / dados[ativos].iloc[0]
    fig = go.Figure()
    for ativo in ativos:
        fig.add_trace(go.Scatter(x=precos_norm.index, y=precos_norm[ativo],
                                 mode='lines', name=ativo))
    fig.update_layout(title=titulo, xaxis_title='Data', yaxis_title='Valor normalizado',
                      hovermode='x unified', height=600)
    return fig

def matriz_correlacao(dados, ativos, titulo="Correlação entre Retornos"):
    retornos = dados[ativos].pct_change().dropna()
    corr = retornos.corr()
    fig, ax = plt.subplots(figsize=(9,7))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, ax=ax)
    ax.set_title(titulo)
    fig.tight_layout()
    return fig

def grafico_risco_retorno(df_metricas, titulo="Risco x Retorno"):
    # espera df_metricas com colunas 'Volatilidade (%)', 'Retorno Anual (%)', 'Sharpe'
    df = df_metricas.copy()
    fig = px.scatter(df.reset_index(), x='Volatilidade (%)', y='Retorno Anual (%)',
                     color='Sharpe', text=df.reset_index().iloc[:,0],
                     color_continuous_scale='RdYlGn', title=titulo)
    fig.update_traces(textposition='top center')
    return fig

def valorizacao_total(dados, ativos, titulo="Valorização Total por Ativo"):
    vals = {}
    for ativo in ativos:
        inicio = dados[ativo].iloc[0]
        fim = dados[ativo].iloc[-1]
        vals[ativo] = ((fim - inicio) / inicio) * 100
    df_val = pd.Series(vals).sort_values().to_frame("Retorno (%)")
    fig = px.bar(df_val, x='Retorno (%)', y=df_val.index, orientation='h', color='Retorno (%)',
                 color_continuous_scale='RdYlGn', title=titulo)
    return fig, df_val

def analise_drawdown(dados, ativos, titulo="Drawdown Máximo por Ativo"):
    precos_norm = dados[ativos] / dados[ativos].iloc[0]
    drawdowns = precos_norm.apply(calcular_drawdown)
    drawdown_max = (drawdowns.min() * 100).round(2).sort_values()
    fig, ax = plt.subplots(figsize=(10,6))
    drawdown_max.plot(kind='barh', ax=ax, color='crimson')
    ax.set_xlabel("Drawdown (%)")
    ax.set_title(titulo)
    fig.tight_layout()
    return fig, drawdowns

def estatisticas_descritivas(dados, ativos):
    stats = pd.DataFrame({
        'Média': dados[ativos].mean(),
        'Desvio Padrão': dados[ativos].std(),
        'Mínimo': dados[ativos].min(),
        'Máximo': dados[ativos].max(),
        'Assimetria': dados[ativos].skew(),
        'Curtose': dados[ativos].kurt()
    }).round(4)
    fig, ax = plt.subplots(figsize=(10,4))
    sns.heatmap(stats.T, annot=True, cmap="YlGnBu", fmt=".4f", ax=ax)
    ax.set_title("Estatísticas Descritivas")
    fig.tight_layout()
    return stats, fig

