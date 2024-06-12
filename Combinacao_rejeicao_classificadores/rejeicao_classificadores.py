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

probabilidade_rede = rede_neural.predict_proba(novo_registro)
confianca_rede = probabilidade_rede.max()

probabilidade_arvore = arvore.predict_proba(novo_registro)
confianca_arvore = probabilidade_arvore.max()

probabilidade_svm = svm.predict_proba(novo_registro)
confianca_svm = probabilidade_svm.max()

print(confianca_rede)
print(confianca_arvore)
print(confianca_svm)

# Agregação das Previsões:
paga, nao_paga = 0, 0
confianca_minima = 0.999999  # Limiar de confiança
algoritmos = 0

if confianca_rede >= confianca_minima:
    algoritmos += 1
    if resposta_rede[0] == 1:
        nao_paga += 1
    else:
        paga += 1

if confianca_arvore >= confianca_minima:
    algoritmos += 1
    if resposta_arvore[0] == 1:
        nao_paga += 1
    else:
        paga += 1

if confianca_svm >= confianca_minima:
    algoritmos += 1
    if resposta_svm[0] == 1:
        nao_paga += 1
    else:
        paga += 1

# Decisão Final:
if paga > nao_paga:
    print(f"Cliente pagará o emprestimo, baseado em {algoritmos}")
elif paga == nao_paga:
    print(f"Empate, baseado em {algoritmos}")
else:
    print(f"CLienta nao pagará o emprestimo, baseado em {algoritmos}")


