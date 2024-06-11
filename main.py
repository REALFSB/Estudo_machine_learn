import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

########################################################################################################################

# Pegando os arquivos
base_credit = pd.read_csv('credit_data.csv')
base_census = pd.read_csv("census.csv")
base_risco_credito = pd.read_csv('risco_credito.csv')

########################################################################################################################

# Pré-Processamento de dados

# Apagando a coluna
# base_credit2 = base_credit.drop('age', axis=1)

# Apagando apenas os registros bugados
# base_credit2 = base_credit.drop(base_credit[base_credit['age'] <= 0].index)

# Pegando a média das idades
idade_media = base_credit['age'][base_credit['age'] > 0].mean()

# Substituindo os valores negativos com a idade média
base_credit.loc[base_credit['age'] <= 0, 'age'] = idade_media.round()

# Substituindo os valores nulos com a idade média
base_credit.fillna(base_credit['age'].mean(), inplace=True)

########################################################################################################################

# Dividindo atributos previsores e classes
x_credit = base_credit.iloc[:, 1:4].values  # Atributos
y_credit = base_credit.iloc[:, 4].values  # Classe

# Fazendo o escalonamento dos previsores
scaler_credit = StandardScaler()  # Instanciando StandarScaler()
x_credit = scaler_credit.fit_transform(x_credit)  # Aplica o escalonamento

########################################################################################################################

# Dividindo atributos previsores e classes
x_census = base_census.iloc[:, 0:14].values  # Atributos
y_census = base_census.iloc[:, 14].values  # Classe

# Fazendo o escalonamento dos previsores
scaler_census = StandardScaler()  # Instanciando StandarScaler()
x_census = scaler_census.fit_transform(x_census)  # Aplica o escalonamento

########################################################################################################################

# Dividindo atributos previsores e classes
x_risco_credito = base_risco_credito.iloc[:, 0:4].values  # Atributos
y_risco_credito = base_risco_credito.iloc[:, 4].values  # Classes

########################################################################################################################

# Categorizando atributos
label_encoder_workclass = LabelEncoder()
label_encoder_education = LabelEncoder()
label_encoder_marital = LabelEncoder()
label_encoder_occupation = LabelEncoder()
label_encoder_relationship = LabelEncoder()
label_encoder_race = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_country = LabelEncoder()

x_census[:, 1] = label_encoder_workclass.fit_transform(x_census[:, 1])
x_census[:, 3] = label_encoder_education.fit_transform(x_census[:, 3])
x_census[:, 5] = label_encoder_marital.fit_transform(x_census[:, 5])
x_census[:, 6] = label_encoder_occupation.fit_transform(x_census[:, 6])
x_census[:, 7] = label_encoder_relationship.fit_transform(x_census[:, 7])
x_census[:, 8] = label_encoder_race.fit_transform(x_census[:, 8])
x_census[:, 9] = label_encoder_sex.fit_transform(x_census[:, 9])
x_census[:, 13] = label_encoder_country.fit_transform(x_census[:, 13])

########################################################################################################################

# Categorizando atributos
label_encoder_historia = LabelEncoder()
label_encoder_divida = LabelEncoder()
label_encoder_garantia = LabelEncoder()
label_encoder_renda = LabelEncoder()

x_risco_credito[:, 0] = label_encoder_historia.fit_transform(x_risco_credito[:, 0])
x_risco_credito[:, 1] = label_encoder_divida.fit_transform(x_risco_credito[:, 1])
x_risco_credito[:, 2] = label_encoder_garantia.fit_transform(x_risco_credito[:, 2])
x_risco_credito[:, 3] = label_encoder_renda.fit_transform(x_risco_credito[:, 3])

########################################################################################################################

# Categorizando com OneHotEncoder
onethorencoder_census = ColumnTransformer(transformers=[
    ('OneHot', OneHotEncoder(),
     [1, 3, 5, 6, 7, 8, 9, 13])],  # Passando as colunas que quero codificar
    remainder='passthrough')

x_census = onethorencoder_census.fit_transform(x_census).toarray()

# Base de treinamento Credit e Census
x_credit_treinamento, x_credit_teste, y_credit_treinamento, y_credit_teste = (
    train_test_split(x_credit, y_credit,
                     test_size=0.25,
                     random_state=0  # Os registros se mantem iguais nas duas bases
                     ))

x_census_treinamento, x_census_teste, y_census_treinamento, y_census_teste = (
    train_test_split(x_census, y_census,
                     test_size=0.15,
                     random_state=0
                     ))

########################################################################################################################

# Salvando as bases de dados com pickle
with open('credit.pkl', mode='wb') as f:
    pickle.dump([x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste], f)

    # with open('census.pkl', mode='wb') as f:
    pickle.dump([x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste], f)

with open('risco_credito.pkl', 'wb') as f:
    pickle.dump([x_risco_credito, y_risco_credito], f)

########################################################################################################################
