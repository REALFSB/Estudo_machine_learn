import pandas as pd
from pyECLAT import ECLAT

# Carregar os dados do arquivo CSV 'mercado.csv' sem cabeçalhos
base_mercado1 = pd.read_csv('../Dados/mercado.csv', header=None)

# Inicializar o algoritmo ECLAT com os dados carregados
eclat = ECLAT(data=base_mercado1)

# Imprimir a DataFrame binária gerada pelo ECLAT (cada coluna representa um item único e cada linha uma transação)
print(eclat.df_bin)

# Imprimir a lista de itens únicos identificados pelo ECLAT
print(eclat.uniq_)

# Ajustar o modelo ECLAT com suporte mínimo de 30%, combinação mínima de 1 item e máxima de 3 itens
indices, suporte = eclat.fit(min_support=0.3, min_combination=1, max_combination=3)

# Imprimir os suportes das combinações de itens encontradas
print(suporte)
