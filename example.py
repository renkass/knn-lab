# coding=utf-8
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from knnLib.KNNClasses import KNNClassifier
from knnLib.KNNClasses import KNNRegressor
from sklearn.utils import shuffle

data = pd.read_csv("xoro1.csv")

feature_names = ['LW', 'LD', 'RW', 'RD']
X = data[feature_names].astype(float).values
y = data['C'].astype(float).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("My Classifier")
classifier = KNNClassifier(20, X_train, y_train)
prediction = classifier.predict(X_test)
print(accuracy_score(y_test, prediction))
print("sklearn Classifier")
classifier = KNeighborsClassifier(n_neighbors=20)
classifier.fit(X_train, y_train)
prediction = classifier.predict(X_test)
print(accuracy_score(y_test, prediction))



# Data Set Information:
#
# Prediction of residuary resistance of sailing yachts at the initial design stage is of a great value for evaluating the shipâ€™s performance and for estimating the required propulsive power. Essential inputs include the basic hull dimensions and the boat velocity.
#
# The Delft data set comprises 308 full-scale experiments, which were performed at the Delft Ship Hydromechanics Laboratory for that purpose.
# These experiments include 22 different hull forms, derived from a parent form closely related to the â€˜Standfast 43â€™ designed by Frans Maas.
#
#
# Attribute Information:
#
# Variations concern hull geometry coefficients and the Froude number:
#
# 1. Longitudinal position of the center of buoyancy, adimensional.
# 2. Prismatic coefficient, adimensional.
# 3. Length-displacement ratio, adimensional.
# 4. Beam-draught ratio, adimensional.
# 5. Length-beam ratio, adimensional.
# 6. Froude number, adimensional.
#
# The measured variable is the residuary resistance per unit weight of displacement:
#
# 7. Residuary resistance per unit weight of displacement, adimensional.


dataRegg = pd.read_csv("Abalon.csv")

dataRegg = shuffle(dataRegg)

data_train = dataRegg.iloc[:250]
data_test = dataRegg.iloc[250:]

X_train = data_train[['lp','pc','ld','bd','lb','fn']].astype(float).values
y_train = data_train[['rr']].astype(float).values

X_test = data_test[['lp','pc','ld','bd','lb','fn']].astype(float).values
y_test = data_test[['rr']].astype(float).values

print("\n")
print("My KNNREgresoor")
regressor = KNNRegressor(2, X_train, y_train)
prediction = regressor.predict(X_test)
print(mean_absolute_error(y_test, prediction))

print("sklearn KNNRegressor")
regressor = KNeighborsRegressor(n_neighbors=2)
regressor.fit(X_train, y_train)
prediction = regressor.predict(X_test)
print(mean_absolute_error(y_test, prediction))











