# read men/women height weight data..
# OBS: Enhed er hhv. pound og inch

import matplotlib.pyplot as plt
import numpy as np

# Load data
data = np.loadtxt('height_weight.csv', delimiter=';', skiprows=1)
x = data[:,1:3]
y = data[:,0]

pound2kg = 0.453592 
inch2cm = 2.54
Xw = pound2kg*X[:,1]
Xh = inch2cm*X[:,0]

print('weight :', np.mean(Xw), 'height:', np.mean(Xh))


#%% input data - scatter plot

idx_men = y == 0
plt.scatter(Xw[idx_men], Xh[idx_men], s=0.1, c='b')
idx_women = y == 1
plt.scatter(Xw[idx_women], Xh[idx_women], s=0.1, c='r')

#%%
from sklearn.linear_model import LogisticRegression
#import matplotlib.cm as cm

print(Xh[idx_men])
XhMen = Xh[idx_men]
XhWomen = Xh[idx_women]
XhMen = XhMen.reshape(-1, 1)
XhWomen = XhWomen.reshape(-1, 1)

XhReshaped = Xh.reshape(-1,1)

logRef = LogisticRegression(random_state=0)
logRef.fit(XhReshaped, y)

y_proba = logRef.predict_proba(XhReshaped)

logPlot = plt.plot(XhReshaped, y_proba[:,1], '.', XhReshaped, y_proba[:,0], '.')
plt.setp(logPlot[0], linewidth=1, color='red', marker='o', markersize=1)
plt.setp(logPlot[1], linewidth=1, color='blue', marker='o', markersize=1)

#fig, ax = plt.subplots()
#CS = ax.contour(Xh, y, logRef)
#ax.clabel(CS, inline=1, fontsize=10)

#%% input-output relation

# køn som funk af w/h
plt.plot(Xw, y, 'r.', markersize=1)
plt.show()
plt.plot(Xh, y, 'r.', markersize=1)
plt.show()

#%% distribution - e.g. weight

plt.hist([Xw[idx_men], Xw[idx_women]], bins=50);





    
