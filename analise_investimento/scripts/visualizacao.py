import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def calcular_drawdown(serie):
    """Calcula drawdown de uma série"""
    topo = serie.cummax()
    return (serie - topo) / topo

def plotar_performance_normalizada(dados, ativos):
    """Plota performance normalizada (base 1.0)"""
    precos_norm = dados[ativos] / dados[ativos].iloc[0]
    
    plt.figure(figsize=(16, 8))
    for ativo in ativos:
        plt.plot(precos_norm.index, precos_norm[ativo], label=ativo, linewidth=2)
    
    plt.title('Performance Normalizada dos Ativos (base 1.0)', fontsize=16)
    plt.xlabel('Data')
    plt.ylabel('Valor Normalizado')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plotar_interativo(dados, ativos, titulo="Performance dos Ativos"):
    """Cria gráfico interativo com Plotly"""
    precos_norm = dados[ativos] / dados[ativos].iloc[0]
    
    fig = go.Figure()
    for ativo in ativos:
        fig.add_trace(go.Scatter(
            x=precos_norm.index,
            y=precos_norm[ativo],
            mode='lines',
            name=ativo
        ))

    fig.update_layout(
        title=titulo,
        xaxis_title='Data',
        yaxis_title='Valor Normalizado',
        hovermode='x unified',
        template='plotly_dark',
        height=600
    )
    fig.show()


def matriz_correlacao(dados, ativos):
    """Plota matriz de correlação dos retornos"""
    retornos = dados[ativos].pct_change().dropna()
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(retornos.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title("Correlação entre os Retornos dos Ativos")
    plt.tight_layout()
    plt.show()


def grafico_risco_retorno(df_metricas):
    """Gráfico scatter Risco x Retorno"""
    df_plot = df_metricas.copy()
    df_plot['Size'] = abs(df_plot['Sharpe']) + 0.1
    
    fig = px.scatter(
        df_plot,
        x='Volatilidade (%)',
        y='Retorno Anual (%)',
        text=df_plot.index,
        color='Sharpe',
        size='Size',
        title='Risco x Retorno dos Ativos',
        color_continuous_scale='RdYlGn',
        size_max=20
    )
    fig.update_traces(textposition='top center')
    fig.update_layout(height=600)
    fig.show()


def valorizacao_total(dados, ativos):
    """Calcula valorização total no período"""
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

def estatisticas_descritivas(dados, ativos):
    """Calcula estatísticas descritivas"""
    stats = pd.DataFrame({
        'Média': dados[ativos].mean(),
        'Desvio Padrão': dados[ativos].std(),
        'Mínimo': dados[ativos].min(),
        'Máximo': dados[ativos].max(),
        'Assimetria': dados[ativos].skew(),
        'Curtose': dados[ativos].kurt()
    }).round(3)
    
    print("ESTATÍSTICAS DESCRITIVAS:")
    print(stats)
    
    # Heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(stats.T, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Estatísticas Descritivas dos Ativos")
    plt.tight_layout()
    plt.show()
    
    return stats