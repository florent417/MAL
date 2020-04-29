# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:23:38 2020

@author: flole
"""

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
X, y = load_iris(return_X_y=True)
clf = LogisticRegression(random_state=0).fit(X, y)
clf.predict(X[:2, :])

clf.predict_proba(X[:2, :])
print(clf.score(X, y))
