# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 14:39:27 2024

@author: TANISHQ
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


dataset = pd.read_csv(r"C:\Users\TANISHQ\Naresh_IT_Everyday_Personal\Artificial Intelligence\4. ANN\Churn Modelling project\Churn_Modelling.csv")


X = dataset.iloc[:,3:-1].values

y = dataset.iloc[:,-1].values


# Lets apply label encoder on gender column
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

X[:,2] = le.fit_transform(X[:,2])


# Lets apply one hot encoding on cities
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# When you use remainder='passthrough' in ColumnTransformer, the one-hot encoded columns are placed first in the transformed dataset, followed by the unchanged columns.

one_hot_columns = ct.transformers_[0][1].get_feature_names_out(input_features=["Geography"])
print(one_hot_columns)

#Scale values of X
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X= sc.fit_transform(X) ################


# Split data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)             


#ANN
ann = tf.keras.models.Sequential()
ann

ann.add(tf.keras.layers.Dense(units=6, activation='relu')) #input layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu')) #hidden layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) #output
 
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


ann.fit(X_train, y_train, epochs=25)


# ann.save(r"C:\Users\TANISHQ\Naresh_IT_Everyday_Personal\Artificial Intelligence\4. ANN\Churn_Modelling project\ann_churn_model.h5")

import pickle
filename = 'scalar.pkl'
with open(filename, 'wb') as file: pickle.dump(sc, file)

import os
os.getcwd()

