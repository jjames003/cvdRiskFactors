#!/usr/bin/env python
# coding: utf-8

# In[15]:


# Import our libraries 

# Pandas and numpy for data wrangling
import pandas as pd
import numpy as np

# Seaborn / matplotlib for visualization 
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Helper function to split our data
from sklearn.model_selection import train_test_split

from sklearn import metrics

# This is our Logit model
from sklearn.linear_model import LogisticRegression

# Helper fuctions to evaluate our model.
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix 

from imblearn.over_sampling import RandomOverSampler

from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


# In[16]:


df = pd.read_csv('heart_data.csv')
df.head()


# In[17]:


df['age'] = df['age'].apply(lambda x: int(x/365))
df.head()


# In[18]:


selected_features = ['age', 'gender', 'height', 'weight', 'ap_hi','ap_lo','cholesterol', 'gluc', 'smoke', 'alco', 'active']

X = df[selected_features]
y = df['cardio']

r=RandomOverSampler()
x_data,y_data=r.fit_resample(X,y)


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=45)


# In[20]:


rf = RandomForestClassifier()
rf.fit(X_train.values, y_train)
y_pred = rf.predict(X_test.values)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)
pred_proba = rf.predict_proba([[55,1,160,68,110,80,5,14,0,1,0]])[:,1]
pred_proba.round(2)


# In[21]:


model = LogisticRegression(max_iter = 10000)
model.fit(X_train.values, y_train)
y_pred = model.predict(X_test.values)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)
pred_proba = model.predict_proba([[55,1,160,68,110,80,5,14,0,1,0]])[:,1]
pred_proba.round(2)


# In[24]:


import pickle
pickle.dump(model, open('model.pkl', 'wb'))


# In[ ]:


model.predict([[55,1,160,68,110,80,5,14,0,1,0]])[:,1]

