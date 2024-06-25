import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

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

# Aplicando o algoritmo DBSCAN:
dbscan_cartao = DBSCAN(eps=0.37, min_samples=5)

# Ajustando o modelo aos dados normalizados
dbscan_cartao.fit_predict(x_cartao)

# Obtendo os rótulos dos clusters identificados pelo DBSCAN
rotulos = dbscan_cartao.labels_

# Criando e exibindo um gráfico de dispersão interativo:
grafico = px.scatter(x=x_cartao[:, 0], y=x_cartao[:, 1], color=rotulos)
grafico.show()
