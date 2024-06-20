import numpy as np
from sklearn.datasets import make_blobs
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Leitura do conjunto de dados
base_cartao = pd.read_csv('../Dados/credit_card_clients.csv', header=1)

# Criação da feature BILL_TOTAL somando os valores das colunas BILL_AMT1 a BILL_AMT6
base_cartao['BILL_TOTAL'] = base_cartao['BILL_AMT1'] + base_cartao['BILL_AMT2'] + base_cartao['BILL_AMT3'] + \
                            base_cartao['BILL_AMT4'] + base_cartao['BILL_AMT5'] + base_cartao['BILL_AMT6']

# Seleção das colunas 1 (LIMIT_BAL) e 25 (BILL_TOTAL) como variáveis de entrada x_cartao
x_cartao = base_cartao.iloc[:, [1, 25]].values

# Normalização dos dados utilizando StandardScaler
scaler_cartao = StandardScaler()
x_cartao = scaler_cartao.fit_transform(x_cartao)

# Lista para armazenar os valores de WCSS (Within-Cluster Sum of Squares)
wcss = []

# Loop para aplicação do algoritmo K-Means com diferentes números de clusters (de 1 a 10)
for i in range(1, 11):
    kmeans_cartao = KMeans(n_clusters=i, random_state=0)
    kmeans_cartao.fit(x_cartao)
    wcss.append(kmeans_cartao.inertia_)

# Aplicação do K-Means com 4 clusters (n_clusters=4)
kmeans_cartao = KMeans(n_clusters=4, random_state=0)

# Atribuição dos rótulos (clusters) para cada amostra no conjunto de dados
rotulos = kmeans_cartao.fit_predict(base_cartao)

# Visualização dos dados utilizando Plotly Express
grafico = px.scatter(x=x_cartao[:, 0], y=x_cartao[:, 1], color=rotulos)
grafico.show()

# Criação de uma lista combinando os dados originais com os rótulos dos clusters
lista_clientes = np.column_stack((base_cartao, rotulos))

# Ordenação da lista de clientes com base nos rótulos dos clusters (coluna 26)
lista_clientes = lista_clientes[lista_clientes[:, 26].argsort()]
