import pickle
import pandas as pd

########################################################################################################################
# Carregamento dos Resultados:
with open('resultado_arvore.pkl', 'rb') as f:
    arvore = pickle.load(f)

with open('resultados_knn.pkl', 'rb') as f:
    knn = pickle.load(f)

with open('resultados_logistica.pkl', 'rb') as f:
    logistica = pickle.load(f)

with open('resultados_svn.pkl', 'rb') as f:
    svn = pickle.load(f)

with open('resultados_redeneural.pkl', 'rb') as f:
    rede = pickle.load(f)

with open('resultados_randomforest.pkl', 'rb') as f:
    randomforest = pickle.load(f)

########################################################################################################################
# Criação do DataFrame:
resultado = pd.DataFrame({'Arvore': arvore,
                          'Knn': knn,
                          'Logistica': logistica,
                          'Svn': svn,
                          'Rede': rede,
                          'RandomForest': randomforest})

########################################################################################################################
# Análise dos Resultados:
print("Todos resultados")
print(resultado)

print("\nDescrição dos resultados")
print(resultado.describe())

print("\nCoeficiente de variação")
print((resultado.std() / resultado.mean()) * 100)
