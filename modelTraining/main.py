#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle


# In[2]:


pickled_model = pickle.load(open('model.pkl', 'rb'))


# In[3]:


"""
Age should be int between 1-100
"""
age = 85

"""
Gender should be int 1 (Male) or 2 (Female)
"""
gender = 1

"""
Height should be between 50-280 cm
"""
height = 100

"""
Weight should be between 15-125 kg
"""
weight = 60

"""
Systolic Blood Pressure should be between 100-200 mm Hg
"""
ap_hi = 150

"""
Diastolic Blood Pressure should be between 70-130 mm Hg
"""
ap_low = 90

"""
Cholesterol has three levels, 1-3. Should be a drop down menu
"""
chol = 1

"""
Glucose levels has three levels, 1-3. Should be a drop down menu
"""
gluc = 1

"""
Smoke, Alcohol, and Active should be are 1 (yes) or 0 (no)
"""
smoke = 0
alco = 0
active = 0


# In[17]:


array = [[age,gender,height,weight,ap_hi,ap_low,chol,gluc,smoke,alco,active]]
pickled_model.predict(array)
per = pickled_model.predict_proba(array)[:,1][0]
per = per.round(4)*100


# In[18]:


print("You have a ",per,"% chance of having cardiovascular disease")

