import pandas as pd
import plotly.express as px
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Carregar os dados
plano_saude = pd.read_csv('../Dados/plano_saude2.csv')

print(plano_saude)

# Separar as variáveis independentes e dependentes
x_plano_saude = plano_saude.iloc[:, 0:1].values
y_plano_saude = plano_saude.iloc[:, 1].values

# Kernel linear
regressor_svr_linear_saude = SVR(kernel='linear')
regressor_svr_linear_saude.fit(x_plano_saude, y_plano_saude)

# Criar gráfico de dispersão para o kernel linear
grafico_linear = px.scatter(x=x_plano_saude.ravel(), y=y_plano_saude)
grafico_linear.add_scatter(x=x_plano_saude.ravel(), y=regressor_svr_linear_saude.predict(x_plano_saude),
                           name='Regressão Linear')
grafico_linear.show()

# Kernel polinomial
regressor_svr_poli_saude = SVR(kernel='poly', degree=2)
regressor_svr_poli_saude.fit(x_plano_saude, y_plano_saude)

# Criar gráfico de dispersão para o kernel polinomial
grafico_poli = px.scatter(x=x_plano_saude.ravel(), y=y_plano_saude)
grafico_poli.add_scatter(x=x_plano_saude.ravel(), y=regressor_svr_poli_saude.predict(x_plano_saude),
                         name='Regressão Polinomial')
grafico_poli.show()

# Escalar as variáveis
scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_plano_saude_scaled = scaler_x.fit_transform(x_plano_saude)
y_plano_saude_scaled = scaler_y.fit_transform(y_plano_saude.reshape(-1, 1))

# Kernel rbf
regressor_svr_rbf_saude = SVR(kernel='rbf')
regressor_svr_rbf_saude.fit(x_plano_saude_scaled, y_plano_saude_scaled.ravel())

# Criar gráfico de dispersão para o kernel rbf
grafico_rbf = px.scatter(x=x_plano_saude_scaled.ravel(), y=y_plano_saude_scaled.ravel())
grafico_rbf.add_scatter(x=x_plano_saude_scaled.ravel(), y=regressor_svr_rbf_saude.predict(x_plano_saude_scaled),
                        name='Regressão RBF')
grafico_rbf.show()
