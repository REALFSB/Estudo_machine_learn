import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Carregar os dados
plano_saude = pd.read_csv('../Dados/plano_saude2.csv')

print(plano_saude)

# Separar as variáveis independentes e dependentes
x_plano_saude = plano_saude.iloc[:, 0:1].values
y_plano_saude = plano_saude.iloc[:, 1].values

# Escalar as variáveis
scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_plano_saude_scaled = scaler_x.fit_transform(x_plano_saude)
y_plano_saude_scaled = scaler_y.fit_transform(y_plano_saude.reshape(-1, 1))

# Criar e treinar o regressor MLP
regressor_rna_saude = MLPRegressor(max_iter=1000)
regressor_rna_saude.fit(x_plano_saude_scaled, y_plano_saude_scaled.ravel())

# Avaliar o desempenho do modelo
print(regressor_rna_saude.score(x_plano_saude_scaled, y_plano_saude_scaled))

# Fazer uma previsão para uma nova idade (novo = [[40]])
novo = [[40]]
novo_scaled = scaler_x.transform(novo)

previsao_scaled = regressor_rna_saude.predict(novo_scaled)
previsao = scaler_y.inverse_transform(previsao_scaled.reshape(-1, 1))
print(previsao)
