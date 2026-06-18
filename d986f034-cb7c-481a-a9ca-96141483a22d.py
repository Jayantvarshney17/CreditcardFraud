"""
## Import Libraries

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
import seaborn as sns

"""
## Choose Dataset from Directory
"""

"""
## Load Dataset

"""

ram = pd.read_csv('creditcard.csv')
ram.head(20)

"""
## Summarize Dataset
"""

ram.shape
ram.describe()

"""
## Check Null values in Dataset
"""

ram.isna().any()

"""
## Find Percentage of Not Fraud transaction Data
"""

"""


1.   dataset[‘Class’] = 0 Not a fraud
2.   dataset[‘Class’] = 1 Fraud



"""

nfcount=0

notFraud=ram['Class']

for i in range(len(notFraud)):

  if notFraud[i]==0:

    nfcount=nfcount+1

nfcount

per_nf=(nfcount/len(notFraud))*100

print('percentage of total not fraud transaction in the dataset: ',per_nf)

"""
## Find Percentage of Fraud transaction Data
"""

fcount=0

Fraud=ram['Class']

for i in range(len(Fraud)):

  if Fraud[i]==1:

    fcount=fcount+1

fcount

per_f=(fcount/len(Fraud))*100

print('percentage of total fraud transaction in the dataset: ',per_f)

"""
## Plot Fraud transaction vs Non-Fraud transaction

Plot Amount Vs Time:
"""

x=ram['Time']

y=ram['Amount']

plt.plot(x, y)

plt.title('Time Vs amount')

plt.show()

"""
## Plot Amount Distribution Curve:
"""

plt.figure(figsize=(10,8), )

plt.title('Amount Distribution')

sns.distplot(ram['Amount'],color='red')

plt.show()

"""
## Correlation between all attribute
"""

correlation_metrics = ram.corr()

fig = plt.figure(figsize = (14, 9))

sns.heatmap(correlation_metrics, vmax = .9, square = True)

plt.show()

x=ram.iloc[:,:-1]#drop the target variable

y=ram['Class']

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 42)

from sklearn.linear_model import LinearRegression

linear = LinearRegression()

linear.fit(xtrain, ytrain)

y_pred = linear.predict(xtest)

table= pd.DataFrame({"Actual":ytest,"Predicted":y_pred})

table
