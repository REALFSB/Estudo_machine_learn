from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

########################################################################################################################

with open('../Dados/credit.pkl', 'rb') as f:
    x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f)

with open('../Dados/census.pkl', 'rb') as f:
    x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste = pickle.load(f)

########################################################################################################################
# Criar e treinar os classificadores Random Forest
random_florest_credit = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
random_florest_credit.fit(x_credit_treinamento, y_credit_treinamento)

random_florest_census = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
random_florest_census.fit(x_census_treinamento, y_census_treinamento)

########################################################################################################################
# Fazer previsões
previsao_credit = random_florest_credit.predict(x_credit_teste)

previsao_census = random_florest_census.predict(x_census_teste)

########################################################################################################################
# Avaliar os resultados
credit_matriz = classification_report(y_credit_teste, previsao_credit)

census_matriz = classification_report(y_census_teste, previsao_census)

########################################################################################################################
# Exibir os resultados
print(credit_matriz)
print(census_matriz)

print("Treinado")
