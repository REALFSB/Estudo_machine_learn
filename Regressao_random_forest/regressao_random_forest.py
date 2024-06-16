import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Carregamento dos dados:
plano_saude = pd.read_csv('../Dados/plano_saude2.csv')

# Exibição dos dados:
print(plano_saude)

# Preparação dos dados para o modelo:
x_plano_saude = plano_saude.iloc[:, 0:1].values
y_plano_saude = plano_saude.iloc[:, 1].values

# Criação do modelo:
regressor_random_forest = RandomForestRegressor(n_estimators=10)

# Treinamento do modelo:
regressor_random_forest.fit(x_plano_saude, y_plano_saude)

# Avaliação do modelo:
print(regressor_random_forest.score(x_plano_saude, y_plano_saude))

# Previsão:
previsao = regressor_random_forest.predict([[40]])
print(previsao)
