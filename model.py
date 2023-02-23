# Import all the required libraries
import gzip
import pickle
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import sklearn
import imblearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from feature_engine.imputation import CategoricalImputer
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from keras import layers

from sklearn.ensemble import RandomForestClassifier

# Load the data
data = pd.read_csv("train.csv")
test = pd.read_csv("test1.csv")

# using sklearn-pandas package
from feature_engine.imputation import CategoricalImputer

# Imputing the categorical variable 'education' using 'frequent' imputation method
datatr = CategoricalImputer(variables=['education'],imputation_method='frequent').fit_transform(data)
datatest= CategoricalImputer(variables=['education'],imputation_method='frequent').fit_transform(test)

# Imputing the numerical variable 'previous_year_rating' using pandas fillna() method
datatr['previous_year_rating'].fillna(datatr['previous_year_rating'].median(), inplace=True)
datatest['previous_year_rating'].fillna(datatest['previous_year_rating'].median(), inplace=True)

datatr = datatr[datatr['length_of_service'] < 13]

# lets create some extra features from existing features to improve our Model

# creating a Metric of Sum
datatr['sum_metric'] = datatr['awards_won?']+ datatr['previous_year_rating']
datatest['sum_metric'] = datatest['awards_won?'] + datatest['previous_year_rating']

# creating a total score column
datatr['total_score'] = datatr['avg_training_score'] * datatr['no_of_trainings']
datatest['total_score'] = datatest['avg_training_score'] * datatest['no_of_trainings']

#Dropping irrelevant fields
datatr = datatr.drop(['recruitment_channel', 'region', 'employee_id'], axis = 1)
datatest = datatest.drop(['recruitment_channel', 'region', 'employee_id'], axis = 1)

#The no. of employee who did not get an award, previous_year_rating as 1 and avg_training score is less than 60 but, still got promotion.
datatr = datatr.drop(datatr[(datatr['previous_year_rating'] == 1.0) & (datatr['awards_won?'] == 0) & (datatr['avg_training_score'] < 60) & (datatr['is_promoted'] == 1)].index)

# Encoding these categorical columns to convert them into numerical columns

# Encode the education in their degree of importance 
datatr['education'] = datatr['education'].replace(("Master's & above", "Bachelor's", "Below Secondary"),
                                                (3, 2, 1))
datatest['education'] = datatest['education'].replace(("Master's & above", "Bachelor's", "Below Secondary"),
                                                (3, 2, 1))

# lets use Label Encoding for Gender and Department to convert them into Numerical
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
datatr['department'] = le.fit_transform(datatr['department'])
datatest['department'] = le.fit_transform(datatest['department'])
datatr['gender'] = le.fit_transform(datatr['gender'])
datatest['gender'] = le.fit_transform(datatest['gender'])

y = datatr['is_promoted']
x = datatr.drop(['is_promoted'], axis = 1)
x_test = datatest

from imblearn.over_sampling import SMOTE

x_resample, y_resample  = SMOTE().fit_resample(x, y.values.ravel())

from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(x_resample, y_resample, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_valid = sc.transform(x_valid)
x_test = sc.transform(x_test)

model = RandomForestClassifier()
model.fit(x_train,y_train)
y_pred = model.predict(x_valid)

pickle.dump(model, open('model.pkl','wb'))
#load the model and test with a custom input
model = pickle.load( open('model.pkl','rb'))

with open('model.pkl', 'rb') as f_in, gzip.open('model.pkl.gz', 'wb') as f_out:
    f_out.write(f_in.read())
