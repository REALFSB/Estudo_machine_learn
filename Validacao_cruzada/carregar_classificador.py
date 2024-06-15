import pickle
import numpy as np

with open('../Dados/credit.pkl', 'rb') as f:
    x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f)

x_credit = np.concatenate((x_credit_treinamento, x_credit_teste), axis=0)
y_credit = np.concatenate((y_credit_treinamento, y_credit_teste), axis=0)

# Carrega os modelos treinados de rede neural, árvore de decisão e SVM a partir de arquivos pickle
rede_neural = pickle.load(open('../Dados/rede_neural_finalizado.sav', 'rb'))
arvore = pickle.load(open('../Dados/arvore_finalizado.sav', 'rb'))
svm = pickle.load(open('../Dados/svm_finalizado.sav', 'rb'))

# Seleciona um novo registro para previsão
novo_registro = x_credit[0]  # Alterar o valor do x_credit

# Ajusta a forma do novo registro para que ele tenha a forma correta para previsão
novo_registro = novo_registro.reshape(1, -1)

# Faz a previsão do novo registro usando a rede neural, árvore de decisão e SVM, e imprime os resultados
print("Neural Network Prediction:", rede_neural.predict(novo_registro))
print("Decision Tree Prediction:", arvore.predict(novo_registro))
print("SVM Prediction:", svm.predict(novo_registro))

