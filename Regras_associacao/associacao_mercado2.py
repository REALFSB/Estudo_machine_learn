import pandas as pd
from apyori import apriori

base_mercado2 = pd.read_csv('../Dados/mercado2.csv', header=None)

# Pré-processamento dos Dados:
transacoes = []
for i in range(base_mercado2.shape[0]):
    transacoes.append([str(base_mercado2.values[i, j]) for j in range(base_mercado2.shape[1])])

# Produtos que são vendidos 4 vezes por dia
# 4 * 7 (dias da semana) / 7501 (quantidade de dados)

# Execução do Algoritmo Apriori:
regras = apriori(transacoes, min_support=0.003, min_confidence=0.2, min_lift=3)
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

# Ajustar a configuração para exibir todas as colunas
pd.set_option('display.max_columns', None)

# Criar o DataFrame com as regras de associação
rules_df = pd.DataFrame({'A': A, 'B': B, 'suporte': suporte, 'confianca': confianca, 'lift': lift})

# Exibir o DataFrame
print(rules_df)