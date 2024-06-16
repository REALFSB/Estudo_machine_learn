import pandas as pd
from apyori import apriori

base_mercado1 = pd.read_csv('../Dados/mercado.csv', header=None)

# Pré-processamento dos Dados:
transacoes = []

for i in range(len(base_mercado1)):
    transacoes.append([str(base_mercado1.values[i, j]) for j in range(base_mercado1.shape[1])])

# Execução do Algoritmo Apriori:
regras = apriori(transacoes, min_support=0.3, min_confidence=0.8, min_lift=2)
resultados = list(regras)

# Extração de Regras de Associação:
r = resultados[2][2]

A = []
B = []

suporte = []
confianca = []
lift = []

for resultado in resultados:
    s = resultado[1]
    result_rules = resultado[2]
    for result_rule in result_rules:
        a = list(result_rule[0])
        b = list(result_rule[1])
        c = result_rule[2]
        l = result_rule[3]
        A.append(a)
        B.append(b)
        suporte.append(s)
        confianca.append(c)
        lift.append(l)

# Criação de um DataFrame para Exibir as Regras:
rules_df = pd.DataFrame({'A': A, 'B': B, 'suporte': suporte, 'confianca': confianca, 'lift': lift})

# Impressão das Regras Ordenadas por Lift:
print(rules_df.sort_values(by='lift', ascending=False))
