from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

########################################################################################################################
# IMPORTAÇÃO DOS ARQUIVOS

with open('../Dados/risco_credito.pkl', 'rb') as f:
    x_risco_credito, y_risco_credito = pickle.load(f)

with open('../Dados/credit.pkl', 'rb') as f:
    x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f)

with open('../Dados/census.pkl', 'rb') as f:
    x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste = pickle.load(f)

########################################################################################################################
# ÁRVORE DE DESCISÃO

# Instanciando o objeto e colocando como critério o modelo de entropia
arvore_risco_credito = DecisionTreeClassifier(criterion='entropy')

# Treinando a árvore
arvore_risco_credito.fit(x_risco_credito, y_risco_credito)

########################################################################################################################
# Instanciando o objeto e colocando como critério o modelo de entropia
arvore_credit = DecisionTreeClassifier(criterion='entropy', random_state=0)

# Treinando a árvore
arvore_credit.fit(x_credit_treinamento, y_credit_treinamento)

########################################################################################################################
# Instanciando o objeto e colocando como critério o modelo de entropia
arvore_census = DecisionTreeClassifier(criterion='entropy')

# Treinando a árvore
arvore_census.fit(x_census_treinamento, y_census_treinamento)

########################################################################################################################
# MÉTODOS ÚTEIS

# Método para ver importancia dos atributos
# arvore_risco_credito.feature_importances_

########################################################################################################################
# FAZENDO A PREVISÃO

# Passando as amostras manualmente
risco_credito_previsao = arvore_risco_credito.predict([[0, 0, 1, 2], [2, 0, 0, 0]])

########################################################################################################################

# Passando as amostras pela variavel
credit_previsao = arvore_credit.predict(x_credit_teste)

########################################################################################################################

# Passando as amostras pela variavel
census_previsao = arvore_census.predict(x_census_teste)

########################################################################################################################
# VERIFICANDO A TAXA DE ACERTOS

# Mostra a taxa de acertos
credit_accuracy = accuracy_score(y_credit_teste, credit_previsao)
print(credit_accuracy)

# Mostra a matriz de confusão
credit_matriz = classification_report(y_credit_teste, credit_previsao)
print(credit_matriz)

########################################################################################################################

# Mostra a taxa de acertos
census_accuracy = accuracy_score(y_census_teste, census_previsao)
print(census_accuracy)

# Mostra a matriz de confusão
census_matriz = classification_report(y_census_teste, census_previsao)
print(census_matriz)

########################################################################################################################

print(risco_credito_previsao, '\n')
print(credit_previsao, '\n')
print(census_previsao)
