# Define the Task
# "Use life expectancy and long-term unemployment rate to predict the perceived happiness (low or high) of inhabitants of a country." 

# Import Python libraries for data manipuation and visualization
import pandas as pd
# import numpy as np
# import matplotlib.pyplot as pyplot

# Import the Python machine learning libraries we need
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Import some convenience functions.  This can be found on the course github
# from functions import *

# Load the data set
dataset = pd.read_csv("world_data_really_tiny.csv")
# print(dataset)

# Inspect first few rows
dataset.head(12)
# print(dataset.head(2))

# Inspect data shape
dataset.shape
# print(dataset.shape)

# Inspect descriptive stats
dataset.describe()
# print(dataset.describe())

# View univariate histgram plots
# histPlotAll(dataset)

# View univariate box plots
# boxPlotAll(dataset)

# View class split
# classComparePlot(dataset[["happiness","lifeexp","unemployment"]], 'happiness', plotType='hist')

# Split into input and output features
y = dataset["happiness"]
X = dataset[["lifeexp","unemployment"]]
X.head()
# print(X.head())
y.head()
# print(y.head())


# Build a model
# 1.Split into test and training sets
# Use the  train_test_split()  function in sklearn to split the sample set into a training set, which we will use to train the model, and a test set, to evaluate the model:

# Split into test and training sets
test_size = 0.33
seed = 7
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=test_size, random_state=seed)

# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

# Select an Algorithm
# 2. Create a model using the sklearn decision tree algorithm:
model = DecisionTreeClassifier()

# Fit model to the data
# Now take the training set and use it to fit the model (i.e., train the model):
model.fit(X_train, y_train)

# Check model performance on training data
# Next, assess how well the model predicts happiness using the training data, by “pouring” training set X into the decision tree:
predictions = model.predict(X_train)
# print(accuracy_score(y_train, predictions))

# Evaluate the model on the test data
predictions = model.predict(X_test)
# print(predictions)
# print(accuracy_score(y_test, predictions))

# We can show the model predictions with the original data, with the actual happiness value:
df = X_test.copy()
df['Actual'] = y_test
df['Prediction'] = predictions
print(df)
