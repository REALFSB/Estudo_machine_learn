import pandas as pd
import plotly.express as px
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression

base_casas = pd.read_csv('../Dados/house_prices.csv')
print(base_casas)
print(base_casas.describe())

x_casas = base_casas.iloc[:, 5:6].values
y_casas = base_casas.iloc[:, 2].values

