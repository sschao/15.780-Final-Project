
# coding: utf-8

# In[46]:

#!/usr/bin/env python3

"""
Created on Wed Dec  5 21:02:20 2018

@author: rosannaz
"""

import csv

# Load the Pandas libraries with alias 'pd' 
import pandas as pd 
# Read data from file 'filename.csv' 
data = pd.read_csv("AllDeals05-18-clean.csv") 
# Preview the first 5 lines of the loaded data 
data.head()


# In[47]:

list(data)


# In[48]:

data = data.dropna()


# In[49]:

data.Status = data.Status.astype('category')
data.Status = data.Status.cat.codes
data.Stage = data.Stage.astype('category')
data.Stage = data.Stage.cat.codes
data.State = data.State.astype('category')
data.State = data.State.cat.codes
data.Name = data.Name.astype('category')
data.Name = data.Name.cat.codes
data.IndustryClassification = data.IndustryClassification.astype('category')
data.IndustryClassification = data.IndustryClassification.cat.codes


# In[50]:

feature_cols = ['Name',
 'Stage',
 'Size',
 'TotalFunding',
 'State',
 'IndustryClassification']


X = data.dropna().loc[:, feature_cols]
y = data.dropna().Status


# In[72]:

y = y.astype('category')
y = y.cat.codes
X.Stage = X.Stage.astype('category')
X.Stage = X.Stage.cat.codes
X.State = X.State.astype('category')
X.State = X.State.cat.codes
X.Name = X.Name.astype('category')
X.Name = X.Name.cat.codes
X.IndustryClassification = X.IndustryClassification.astype('category')
X.IndustryClassification = X.IndustryClassification.cat.codes


# In[51]:

from sklearn.model_selection import train_test_split, GridSearchCV
# Partition data set into training/test split (2 to 1 ratio)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/10., random_state=42)


# In[52]:

from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100.)


# In[53]:

clf.fit(X_train, y_train)


# In[54]:

clf.predict(X_test)


# In[55]:

predict = list(clf.predict(X_test))


# In[56]:

y_p = list(y_test)


# In[85]:

count = 0
for i in range(len(y_p)):
    if predict[i] == y_p[i]:
        count+=1


# In[86]:

count


# In[92]:

X_test


# In[57]:

clf.score(X_test, y_test)    


# In[88]:

from sklearn.metrics import accuracy_score
accuracy_score(y_p, predict)


# In[62]:

import numpy as np
data['Date']=data['Date'].astype(str).str[-2:].astype(np.int64)
data.Date = data.Date.astype(int)

#In [11]: df = pd.DataFrame(np.random.randn(100, 2))

#In [12]: msk = np.random.rand(len(df)) < 0.8

#In [13]: train = df[msk]

#In [14]: test = df[~msk]


# In[74]:

data_p = data.loc[data['Date'].between(4, 10)]
Xp = data_p.dropna().loc[:, feature_cols]
yp = data_p.dropna().Status
p = clf.predict(Xp)


# In[79]:

Xp['Predict']=p
Xp['Status']=yp
Xp['Date']= data_p.Date
X_port = Xp.loc[Xp['Status']==1]


# In[80]:

X_port.sample(50)


# In[68]:

clf.score(Xp, yp)    

