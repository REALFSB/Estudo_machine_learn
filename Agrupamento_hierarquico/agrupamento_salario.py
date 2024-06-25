import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import plotly.express as px

# Definindo listas de valores para x (idades) e y (salários)
x = [20, 27, 21, 25, 30, 32, 28, 24, 26, 29, 23, 22, 31, 19, 33]
y = [1000, 1200, 1900, 1500, 1700, 2000, 2100, 1800, 1600, 1750, 1600, 1400, 1950, 1100, 2050]

# Criando uma matriz 2D com as listas x e y
base_salario = np.array(list(zip(x, y)))

# Normalização:
scaler_salario = StandardScaler()

# Ajustando e transformando a base de dados para normalizar
base_salario = scaler_salario.fit_transform(base_salario)

# Criando e exibindo o dendrograma:
dendograma = dendrogram(linkage(base_salario, method='ward'))

# Configurando e exibindo o título e os rótulos do dendrograma
plt.title('Dendograma')
plt.xlabel('Pessoas')
plt.ylabel('Distância')
plt.show()

# Realizando a clusterização aglomerativa:
hc_salario = AgglomerativeClustering(n_clusters=3, linkage='ward')

# Ajustando o modelo e obtendo os rótulos dos clusters
rotulos = hc_salario.fit_predict(base_salario)

# Criando e exibindo um gráfico de dispersão interativo:
grafico = px.scatter(x=base_salario[:, 0], y=base_salario[:, 1], color=rotulos)

# Exibindo o gráfico interativo
grafico.show()
