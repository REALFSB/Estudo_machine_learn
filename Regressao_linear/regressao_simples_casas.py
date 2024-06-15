import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Carregando a base de dados de preços de casas a partir de um arquivo CSV
base_casas = pd.read_csv('../Dados/house_prices.csv')

# Exibindo o DataFrame completo
print(base_casas)

# Exibindo estatísticas descritivas do DataFrame
print(base_casas.describe())

# Extraindo a variável independente (coluna 5) e a variável dependente (coluna 2)
x_casas = base_casas.iloc[:, 5:6].values
y_casas = base_casas.iloc[:, 2].values

# Dividindo os dados em conjuntos de treinamento (70%) e teste (30%)
x_casas_treinamento, x_casas_teste, y_casas_treinamento, y_casas_teste = train_test_split(
    x_casas, y_casas, test_size=0.3, random_state=0)

# Criando um modelo de regressão linear
regressor_simples_casas = LinearRegression()

# Ajustando o modelo aos dados de treinamento
regressor_simples_casas.fit(x_casas_treinamento, y_casas_treinamento)

# Imprimindo o intercepto do modelo (b0)
print(regressor_simples_casas.intercept_)

# Imprimindo o coeficiente do modelo (b1)
print(regressor_simples_casas.coef_)

# Imprimindo o coeficiente de determinação (R^2) do modelo nos dados de treinamento
print(regressor_simples_casas.score(x_casas_treinamento, y_casas_treinamento))

# Imprimindo o coeficiente de determinação (R^2) do modelo nos dados de teste
print(regressor_simples_casas.score(x_casas_teste, y_casas_teste))

# Realizando previsões com o modelo ajustado nos dados de treinamento
previsoes = regressor_simples_casas.predict(x_casas_treinamento)

# Criando um gráfico de dispersão dos dados de treinamento
grafico1 = px.scatter(x=x_casas_treinamento.ravel(), y=y_casas_treinamento)

# Criando uma linha de previsão usando os dados de treinamento
grafico2 = px.line(x=x_casas_treinamento.ravel(), y=previsoes)

# Alterando a cor da linha de previsão para vermelho
grafico2.data[0].line.color = 'red'

# Combinando os dois gráficos (dispersão e linha de previsão) em uma única figura
grafico3 = go.Figure(data=grafico1.data + grafico2.data)

# Exibindo o gráfico combinado
grafico3.show()
