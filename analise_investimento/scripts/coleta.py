import yfinance as yf
import sgs
import pandas as pd
from datetime import datetime

def ler_tickers(arquivo="tickers_populares.txt"):
    tickers = []
    with open(arquivo, "r") as f:
        for linha in f:
            linha = linha.strip() 
            if linha and not linha.startswith("#"):
                tickers.append(linha)
    return tickers


class ColetorDados:
    def __init__(self):
        self.ativos_yf = []
        self.criptos = []
        self.dados = pd.DataFrame()
    
    def adicionar_ativos(self, tickers):
        if isinstance(tickers, str):
            tickers = [tickers]
        self.ativos_yf.extend(tickers)
        return self 
        
    def adicionar_criptos(self, criptos):
        if isinstance(criptos, str):
            criptos = [criptos]
        self.criptos.extend(criptos)
        return self

    def adicionar_tickers_de_arquivo(self, arquivo="tickers.txt"):
        adicionando_criptos = False
        with open(arquivo, "r") as f:
            for linha in f:
                linha = linha.strip()
                if not linha or linha.startswith("#"):
                    # Detecta seção de criptos
                    if linha.upper() == "#CRIPTOS":
                        adicionando_criptos = True
                    continue
                if adicionando_criptos:
                    self.adicionar_criptos(linha)
                else:
                    self.adicionar_ativos(linha)
        return self
    
    def carregar_tickers_txt(caminho="tickers.txt"):
        acoes, criptos = [], []
        with open(caminho, "r", encoding="utf-8") as f:
            for linha in f:
                t = linha.strip()
                if not t or t.startswith("#"):  # ignora vazio/comentários
                    continue
                # Criptos (sem .SA e geralmente 3-5 letras ou stablecoins)
                if t.upper() in ["BTC","ETH","BNB","ADA","SOL","XRP","DOGE","DOT","MATIC","AVAX","LINK","ATOM","LTC","TRX","UNI","XLM","BCH","USDT","USDC"]:
                    criptos.append(t)
                else:
                    acoes.append(t)
        return acoes, criptos


    def coletar_todos(self, start, end=None):
        if end is None:
            end = datetime.today().strftime('%Y-%m-%d')

        dados_coletados = []
        
        if self.ativos_yf:
            print(f"📈 Coletando {len(self.ativos_yf)} ativos: {', '.join(self.ativos_yf)}")
            try:
                dados_yf = baixar_dados_yf(self.ativos_yf, start, end)
                if not dados_yf.empty:
                    dados_coletados.append(dados_yf)
                    print(f"Ativos coletados: {dados_yf.shape}")
                else:
                    print("Nenhum dado de ativo coletado")
            except Exception as e:
                print(f"Erro ao coletar ativos: {e}")
        
        for cripto in self.criptos:
            print(f"₿ Coletando {cripto}...")
            try:
                cripto_data = pegar_preco_cripto(cripto, start=start, end=end)
                if not cripto_data.empty:
                    dados_coletados.append(cripto_data)
                    print(f"✅ {cripto} coletado: {cripto_data.shape}")
                else:
                    print(f"⚠️ Nenhum dado para {cripto}")
            except Exception as e:
                print(f"❌ Erro ao coletar {cripto}: {e}")

        
        if dados_coletados:
            print("🔗 Juntando todos os dados...")
            self.dados = dados_coletados[0].copy()
            
            for df in dados_coletados[1:]:
                for col in df.columns:
                    if col in self.dados.columns:
                        print(f"⚠️ Coluna duplicada encontrada: {col}. Pulando para evitar conflito.")
                        df = df.drop(columns=[col])
                self.dados = self.dados.join(df, how='outer')
            
            self.dados = self.dados.sort_index().ffill()
            
            print(f"🎉 Coleta finalizada! Dataset final: {self.dados.shape}")
            print(f"📅 Período: {self.dados.index.min()} até {self.dados.index.max()}")
            print(f"📊 Colunas: {list(self.dados.columns)}")
        else:
            print("❌ Nenhum dado foi coletado!")
            self.dados = pd.DataFrame()
        
        return self.dados
    
    def obter_dados(self):
        return self.dados
    
    def limpar(self):
        self.ativos_yf = []
        self.criptos = []
        self.dados = pd.DataFrame()
        return self
        
    def resumo(self):
        print("📋 CONFIGURAÇÃO ATUAL:")
        print(f"  📈 Ativos YF: {self.ativos_yf}")
        print(f"  ₿ Criptos: {self.criptos}")
        if not self.dados.empty:
            print(f"  📊 Dados coletados: {self.dados.shape}")
        else:
            print("  📊 Nenhum dado coletado ainda")
            
    def __repr__(self):
        return f"ColetorDados(ativos={len(self.ativos_yf)}, criptos={len(self.criptos)}, dados={self.dados.shape})"
    

def baixar_dados_yf(tickers, start, end=None):
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
    df = yf.download(tickers, start=start, end=end)
    print("Colunas disponíveis:", df.columns.tolist())
    print("Tipo de colunas:", type(df.columns))
    
    if isinstance(df.columns, pd.MultiIndex):
        print("Níveis do MultiIndex:", df.columns.levels)
        if 'Adj Close' in df.columns.levels[0]:
            return df['Adj Close']
        elif 'Close' in df.columns.levels[0]:
            return df['Close']
    else:
        if 'Adj Close' in df.columns:
            return df['Adj Close']
        elif 'Close' in df.columns:
            return df['Close']
    
    return df

def pegar_preco_cripto(cripto_nome, start, end=None):
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
    
    cripto_map = {
        'bitcoin': 'BTC-USD',
        'ethereum': 'ETH-USD',
        'cardano': 'ADA-USD',
        'solana': 'SOL-USD',
        'litecoin': 'LTC-USD',
        'chainlink': 'LINK-USD',
        'polkadot': 'DOT-USD'
    }

    ticker = cripto_map.get(cripto_nome.lower(), f"{cripto_nome.upper()}-USD")

    try:
        df = yf.download(ticker, start=start, end=end)
        if df.empty:
            print(f"Sem dados para {ticker}")
            return pd.Series(name=ticker)

        serie = df['Close'].copy()
        serie.name = ticker
        return serie


    except Exception as e:
        print(f"Erro ao buscar {cripto_nome}: {e}")
        return pd.Series(name=cripto_nome.lower())


