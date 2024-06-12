import pickle
import numpy as np


# Carregamento dos Dados:
with open('../Dados/credit.pkl', 'rb') as f:
    x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f)

# Concatenar Dados de Treinamento e Teste:
x_credit = np.concatenate((x_credit_treinamento, x_credit_teste), axis=0)
y_credit = np.concatenate((y_credit_treinamento, y_credit_teste), axis=0)

# Carregamento dos Modelos Treinados:
rede_neural = pickle.load(open('../Dados/rede_neural_finalizado.sav', 'rb'))
arvore = pickle.load(open('../Dados/arvore_finalizado.sav', 'rb'))
svm = pickle.load(open('../Dados/svm_finalizado.sav', 'rb'))

# Preparação do Novo Registro:
novo_registro = x_credit[1999]

# Ajusta a forma do novo registro para que ele tenha a forma correta para previsão
novo_registro = novo_registro.reshape(1, -1)

# Previsão com os Modelos:
resposta_rede = rede_neural.predict(novo_registro)
resposta_arvore = arvore.predict(novo_registro)
resposta_svm = svm.predict(novo_registro)

# Agregação das Previsões:
paga, nao_paga = 0, 0

if resposta_rede[0] == 1:
    nao_paga += 1
else:
    paga += 1

if resposta_arvore[0] == 1:
    nao_paga += 1
else:
    paga += 1

if resposta_svm[0] == 1:
    nao_paga += 1
else:
    paga += 1

# Decisão Final:
if paga > nao_paga:
    print("Cliente pagará o emprestimo")
elif paga == nao_paga:
    print("Empate")
else:
    print("CLienta nao pagará o emprestimo")


