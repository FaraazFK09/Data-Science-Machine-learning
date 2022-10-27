#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[3]:


diabetes_dataset = pd.read_csv('diabetes.csv')


# In[4]:


diabetes_dataset.head()


# In[5]:


diabetes_dataset.shape


# In[6]:


diabetes_dataset.isnull()


# In[7]:


diabetes_dataset.describe()


# In[9]:


diabetes_dataset['Outcome'].value_counts()


# In[10]:


diabetes_dataset.groupby('Outcome').mean()


# In[12]:


X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']


# In[13]:


print(X)


# In[14]:


print(Y)


# In[21]:


standardizeddata = StandardScaler().fit_transform(X)


# In[22]:


standardizeddata


# In[23]:


X = standardizeddata


# In[24]:


# splitting data


# In[25]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)


# In[26]:


print(X.shape, X_train.shape, X_test.shape)


# In[27]:


#training the model SUPPORT VECTOR MACHINE


# In[29]:


classifier = svm.SVC(kernel='linear')


# In[30]:


classifier.fit(X_train, Y_train)


# In[31]:


X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[32]:


print('Accuracy score of the training data : ', training_data_accuracy)


# In[ ]:




