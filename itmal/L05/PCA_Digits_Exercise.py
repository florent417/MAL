# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 11:07:54 2020

@author: flole
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

#%%
imgNbr = 0
originalDim = 8

X_digits, y_digits = load_digits(return_X_y=True)
digits = load_digits()

plt.imshow(X_digits[imgNbr].reshape(originalDim,originalDim),cmap='binary')
plt.show()
plt.imshow(digits.images[imgNbr], cmap="binary")
plt.show()

#plt.scatter()
print(y_digits[imgNbr])

#%%
from sklearn.decomposition import PCA
# Tror at den inddeler det i farver
originalComps = 64
pca = PCA(n_components=originalComps)

X2D = pca.fit_transform(digits.data)

plt.imshow(X2D[imgNbr].reshape(originalDim,originalDim),cmap='binary')
plt.show()

#%% 
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
print(pca.n_components_)
dimensions = np.arange(pca.n_components_)

# To get to 1.0 probability
cumsum = np.cumsum(pca.explained_variance_ratio_)

plt.plot(dimensions, cumsum)
plt.show()
#%%
import math
numComp = 25
dim = int(math.sqrt(numComp))
print(cumsum[numComp])
# Valg af antal komponenter falder på 25, da dette inkluderer 
# 93% af billederne, eller man kan få det konstrueret med 93% sammenligning
pca25 = PCA(n_components=numComp)
digits_25 = pca25.fit_transform(digits.data)

plt.imshow(digits_25[imgNbr].reshape(-dim,dim), cmap="binary")
plt.show()

digits_25_recovered = pca25.inverse_transform(digits_25)
plt.imshow(digits_25_recovered[imgNbr].reshape(-originalDim,originalDim),cmap='binary')
plt.show()

# Kan ikke forstå opg med hvor meget data fylder jeres billeder før og efter
#%% How to get where the numbers are equal to the mean?
print(np.mean(y_digits))
y_digits_mean = (y_digits == 4)

#%%
