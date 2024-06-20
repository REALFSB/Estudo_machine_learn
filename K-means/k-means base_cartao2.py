import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

# Leitura do conjunto de dados
base_cartao = pd.read_csv('../Dados/credit_card_clients.csv', header=1)

# Criação da feature BILL_TOTAL somando os valores das colunas BILL_AMT1 a BILL_AMT6
base_cartao['BILL_TOTAL'] = base_cartao['BILL_AMT1'] + base_cartao['BILL_AMT2'] + base_cartao['BILL_AMT3'] + \
                            base_cartao['BILL_AMT4'] + base_cartao['BILL_AMT5'] + base_cartao['BILL_AMT6']

# Seleção das colunas 1 (LIMIT_BAL) e 25 (BILL_TOTAL) como variáveis de entrada x_cartao_mais
x_cartao_mais = base_cartao.iloc[:, [1, 2, 3, 4, 5, 25]].values

# Normalização dos dados utilizando StandardScaler
scaler_cartao_mais = StandardScaler()
x_cartao_mais = scaler_cartao_mais.fit_transform(x_cartao_mais)

# Lista para armazenar os valores de WCSS (Within-Cluster Sum of Squares)
wcss = []

# Loop para aplicação do algoritmo K-Means com diferentes números de clusters (de 1 a 10)
for i in range(1, 11):
    kmeans_cartao_mais = KMeans(n_clusters=i, random_state=0)
    kmeans_cartao_mais.fit(x_cartao_mais)
    wcss.append(kmeans_cartao_mais.inertia_)

# Aplicação do K-Means com 4 clusters (n_clusters=4)
kmeans_cartao_mais = KMeans(n_clusters=4, random_state=0)

# Atribuição dos rótulos (clusters) para cada amostra no conjunto de dados x_cartao_mais
rotulos = kmeans_cartao_mais.fit_predict(x_cartao_mais)

# Redução de dimensionalidade utilizando PCA para visualização em 2D
pca = PCA(n_components=2)
x_cartao_mais_pca = pca.fit_transform(x_cartao_mais)

# Visualização dos dados reduzidos utilizando Plotly Express
grafico = px.scatter(x=x_cartao_mais_pca[:, 0], y=x_cartao_mais_pca[:, 1], color=rotulos)
grafico.show()

# Criação de uma lista combinando os dados originais com os rótulos dos clusters
lista_clientes = np.column_stack((base_cartao, rotulos))

# Ordenação da lista de clientes com base nos rótulos dos clusters (coluna 26)
lista_clientes = lista_clientes[lista_clientes[:, 26].argsort()]
