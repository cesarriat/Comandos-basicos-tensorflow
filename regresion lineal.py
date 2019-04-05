
# coding: utf-8

# In[3]:


from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")

california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe["median_house_value"] /= 1000.0
california_housing_dataframe
california_housing_dataframe.describe()

# Primer Paso: Definir las características y configurar las denominadas columnas de características

# Definir la característica de entrada: total_rooms.
my_feature = california_housing_dataframe [["total_rooms"]]

# Configurar una columna numérica de característica para total_rooms.
feature_columns = [tf.feature_column.numeric_column("total_rooms")]

# Segundo Paso : Definir el Objetivo (Target)

# Definir la etiqueta.
targets = california_housing_dataframe["median_house_value"]

# Tercer Paso: Configurar el LinearRegressor

# Usar descenso de gradiente como el optimizador para entrenar el modelo.
my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

# Configurar el modelo de regresión lineal con nuestras columnas característica y optimizador.
# Configurar una tasa de aprendizaje de 0.0000001 para Descenso de Gradiente.
linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=feature_columns,
    optimizer=my_optimizer
)






















# In[ ]:




