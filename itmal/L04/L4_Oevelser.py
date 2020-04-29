# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:45:10 2020

@author: flole
"""
#%% Information
# Read California Housing data
# https://scikit-learn.org/dev/datasets/index.html#california-housing-dataset
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html

# tips https://stackoverflow.com/questions/54925458/what-is-outline-explorer-in-spyder/54925573#54925573

#%% Imports
from sklearn.datasets.california_housing import fetch_california_housing
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

#%% Oevelse 1 a Show example data
cal_housing = fetch_california_housing()
# Load the median house set dataset, y is in 100.000 units
X, avg_house_vals_y = cal_housing.data, cal_housing.target
names = cal_housing.feature_names

Nsamples = X.shape[0]
# arbitrary nbr
n = 42
xn = X[n, :]

# Input values
for i,val in enumerate(xn):
    print(names[i], val)

# Target value
print('Target :', avg_house_vals_y[n])

#%% 1D data plot

plt.plot(X[:,0], avg_house_vals_y, '.', markersize=1)
plt.title(names[0])

#%% Oevelse 1b 
med_house_features_X= X[:,0]
med_house_features_X = med_house_features_X.reshape(-1,1)

# create lin regression obj
regr = linear_model.LinearRegression()

# Split the data sets into training sets and test sets
med_house_feats_X_train = med_house_features_X[:-2100] 
med_house_feats_X_test = med_house_features_X[-2100:]

avg_house_vals_y_train = avg_house_vals_y[:-2100]
avg_house_vals_y_test = avg_house_vals_y[-2100:]

# Train the model using the training sets
regr.fit(med_house_feats_X_train, avg_house_vals_y_train)

# Make predictions using the testing set
med_house_feats_y_pred = regr.predict(med_house_feats_X_test)
#%% Coefficients
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(avg_house_vals_y_test, med_house_feats_y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(avg_house_vals_y_test, med_house_feats_y_pred))
#%% plots
# Dont know whats going on, ask about this

plt.title(names[0] + " test values")
plt.plot(med_house_feats_X_test, avg_house_vals_y_test, '.', markersize=1)
plt.show()

plt.plot(med_house_feats_X_train, avg_house_vals_y_train, '.', markersize=1)
plt.title(names[0] + " train values")
plt.show()

plt.plot(med_house_feats_y_pred, avg_house_vals_y_test, '.', markersize=1)
plt.title(names[0] + " predicted values")
plt.show()
#%% Histograms

bins_edges = np.arange(0, 9, 1.5)
title = "Histogram "

plt.hist(med_house_feats_y_pred, bins=bins_edges)
plt.title(title + " predicted values")
plt.show()

plt.hist(med_house_features_X, bins=bins_edges)
plt.title(title + " original values")
plt.show()

plt.hist(med_house_feats_X_train, bins=bins_edges)
plt.title(title + " train values")
plt.show()

plt.hist(med_house_feats_X_test, bins=bins_edges)
plt.title(title + " test values")
plt.show()

#%% Oevelse 3 How??
def splitToSets(x):
    return x[:-2100], x[-2100:]

# Split the data sets into training sets and test sets

X_1_train, X_0_test = splitToSets(X[:,0])
X_2_train, X_0_test = splitToSets(X[:,0])
X_3_train, X_0_test = splitToSets(X[:,0])
X_4_train, X_0_test = splitToSets(X[:,0])
X_5_train, X_0_test = splitToSets(X[:,0])
X_6_train, X_0_test = splitToSets(X[:,0])
X_7_train, X_0_test = splitToSets(X[:,0])
X_8_train, X_0_test = splitToSets(X[:,0])
 
med_house_X_test = X[-2100:]
#%%
# Train the model using the training sets
regr.fit(X, avg_house_vals_y)

# Make predictions using the testing set
med_house_feats_y_pred = regr.predict(med_house_feats_X_test)
#%% Coefficients
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(avg_house_vals_y_test, med_house_feats_y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(avg_house_vals_y_test, med_house_feats_y_pred))
