import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import plotly.express as px

plano_saude = pd.read_csv('../Dados/plano_saude2.csv')

print(plano_saude)

x_plano_saude = plano_saude.iloc[:, 0:1].values
y_plano_saude = plano_saude.iloc[:, 1].values
print(x_plano_saude)

# Cria e treina o modelo de regressão com árvore de decisão
regressor_arvore_saude = DecisionTreeRegressor()
regressor_arvore_saude.fit(x_plano_saude, y_plano_saude)

# Faz previsões com os dados de entrada
previsoes = regressor_arvore_saude.predict(x_plano_saude)

# Cria uma grade de valores para uma linha de regressão mais suave
x_teste_arvore = np.arange(min(x_plano_saude), max(x_plano_saude), 0.1)
x_teste_arvore = x_teste_arvore.reshape(-1, 1)

# Cria um gráfico de dispersão dos dados reais e adiciona a linha de regressão
grafico = px.scatter(x=x_plano_saude.ravel(), y=y_plano_saude)
grafico.add_scatter(x=x_teste_arvore.ravel(), y=regressor_arvore_saude.predict(x_teste_arvore), name='Regressão')
grafico.show()

# Teste
print(regressor_arvore_saude.predict([[40]]))
