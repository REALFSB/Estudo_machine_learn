import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression

# Carregando a base de dados
base_plano_saude = pd.read_csv('../Dados/plano_saude.csv')

# Extraindo as variáveis independentes e dependentes
x_plano_saude = base_plano_saude.iloc[:, 0].values
y_plano_saude = base_plano_saude.iloc[:, 1].values

# Calculando a correlação
print(np.corrcoef(x_plano_saude, y_plano_saude))

# Ajustando o reshape para o formato correto
x_plano_saude = x_plano_saude.reshape(-1, 1)

# Criando o regressor e ajustando o modelo
regressor_plano_saude = LinearRegression()
regressor_plano_saude.fit(x_plano_saude, y_plano_saude)

# Imprimindo o intercepto e o coeficiente
print(regressor_plano_saude.intercept_)
print(regressor_plano_saude.coef_)

# Realizando previsões com o modelo ajustado
previsoes = regressor_plano_saude.predict(x_plano_saude)

# Imprimindo as previsões
print(previsoes)

# Criando um gráfico de dispersão dos dados originais
grafico = px.scatter(x=x_plano_saude.ravel(), y=y_plano_saude)

# Adicionando a linha de regressão ao gráfico
grafico.add_scatter(x=x_plano_saude.ravel(), y=previsoes, name="Regressão")

# Exibindo o gráfico
grafico.show()

# Calculando a previsão para um novo valor de x (neste caso, 18)
print(regressor_plano_saude.intercept_ + regressor_plano_saude.coef_ * 18)

# Imprimindo o coeficiente de determinação (R^2) do modelo
print(regressor_plano_saude.score(x_plano_saude, y_plano_saude))
