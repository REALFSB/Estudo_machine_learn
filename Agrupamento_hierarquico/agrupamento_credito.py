import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

# Leitura do conjunto de dados
base_cartao = pd.read_csv('../Dados/credit_card_clients.csv', header=1)

# Criação da feature BILL_TOTAL somando os valores das colunas BILL_AMT1 a BILL_AMT6
# Criando uma nova coluna BILL_TOTAL que é a soma das colunas BILL_AMT1 até BILL_AMT6
base_cartao['BILL_TOTAL'] = base_cartao['BILL_AMT1'] + base_cartao['BILL_AMT2'] + base_cartao['BILL_AMT3'] + \
                            base_cartao['BILL_AMT4'] + base_cartao['BILL_AMT5'] + base_cartao['BILL_AMT6']

# Seleção das colunas 1 (LIMIT_BAL) e 25 (BILL_TOTAL) como variáveis de entrada x_cartao
# Selecionando as colunas LIMIT_BAL (coluna 1) e BILL_TOTAL (coluna 25) para a análise
x_cartao = base_cartao.iloc[:, [1, 25]].values

# Normalização dos dados utilizando StandardScaler
scaler_cartao = StandardScaler()

# Ajustando e transformando os dados para normalizar
x_cartao = scaler_cartao.fit_transform(x_cartao)

# Criando e exibindo o dendrograma:
dendograma = dendrogram(linkage(x_cartao, method='ward'))

# Configurando e exibindo o título e os rótulos do dendrograma
plt.title('Dendograma')

# ATENÇÃO! POR TER MUITOS DADOS, A GERAÇÃO DO DENDOGRAMA PODE DEMORAR
# plt.show()

# Realizando a clusterização aglomerativa:
hc_cartao = AgglomerativeClustering(n_clusters=3, linkage='ward')

# Ajustando o modelo e obtendo os rótulos dos clusters
rotulos = hc_cartao.fit_predict(x_cartao)

# Criando e exibindo um gráfico de dispersão interativo:
grafico = px.scatter(x=x_cartao[:, 0], y=x_cartao[:, 1], color=rotulos)
grafico.show()
