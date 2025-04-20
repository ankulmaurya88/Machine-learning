# Build and evaluate a regression model

# Core libraries
import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# Sklearn functionality
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_absolute_error

# Convenience functions.  This can be found on the course github
# from functions import *

# Requirements - "Make predictions about a country's life expectancy in years from a set of metrics for the country."

# 1. Load the data set
dataset = pd.read_csv("world_data.csv")
print(dataset)
# Remove sparsely populated features
dataset = dataset.drop(["murder","urbanpopulation","unemployment"], axis=1)

# Impute all features with mean
means = dataset.mean().to_dict()
# print(means)

for m in means:
    dataset[m] = dataset[m].fillna(value=means[m])

# select features /prepare data
dataset.columns
# print(dataset.columns)

y = dataset["lifeexp"]
X = dataset[['happiness', 'income', 'sanitation', 'water', 'literacy', 'inequality', 'energy', 'childmortality', 'fertility',  'hiv', 'foodsupply', 'population']]

# scale features
# Rescale the data
scaler = MinMaxScaler(feature_range=(0,1))
rescaledX = scaler.fit_transform(X)

# Convert X back to a Pandas DataFrame, for convenience
X = pd.DataFrame(rescaledX, index=X.index, columns=X.columns)
# print(X)

# Build models
# Split into test and training sets
test_size = 0.33
seed = 1
X_train, X_test, Y_train, Y_test =  train_test_split(X, y, test_size=test_size, random_state=seed)

# Create multiple models, fit them and check them
# Create and check a number of models
models = [LinearRegression(), KNeighborsRegressor(), SVR(gamma='auto')]

for model in models:
    
    model.fit(X_train, Y_train)
    predictions = model.predict(X_train)
    # print(type(model).__name__, mean_absolute_error(Y_train, predictions))

# Output
# LinearRegression 2.2920035925091766
# KNeighborsRegressor 2.1955055341375442
# SVR 3.6117510998705655

# Evaluate the models
for model in models:
    predictions = model.predict(X_test)
    # print(type(model).__name__, mean_absolute_error(Y_test, predictions))

# Choose best model
model = models[0]

# See predictions made
predictions = model.predict(X_test)
df = X_test.copy()
df['Prediction'] = predictions
df['Actual'] = Y_test
df["Error"] = Y_test - predictions
print(df)

# Interpret model¶
# Intepret linear regression model
models[0].coef_

models[0].intercept_
