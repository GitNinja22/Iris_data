#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns


# In[2]:


df = pd.read_csv('Iris.csv')
df.head()


# In[3]:


df = df.drop(columns=['Id'])
df


# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


df['Species'].value_counts


# In[7]:


#check for null values
df.isnull().sum()


# In[8]:


df['SepalLengthCm'].hist()


# In[ ]:


df['PetalWidthCm'].hist()


# In[11]:


#scatter plot
colors = ['red','orange','blue']
species = ['Iris-setosa','Iris-virginica','Iris-versicolor']


# In[12]:


for i in range(3):
    x = df[df['Species']== species[i]]
    plt.scatter(x['SepalLengthCm'],x['SepalWidthCm'], c = colors[i],label = species[i])
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()


# In[10]:


df.corr()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[ ]:


df['Species'] = le.fit_transform(df['Species'])
df.head()


# MODEL TRAINING

# In[13]:


from sklearn.model_selection import train_test_split
#train = 70
#test = 30
X = df.drop(columns=['Species'])
Y = df['Species']
x_train,x_test,y_train,y_test = train_test_split(X, Y,test_size = 0.30)
print(y_train)


# In[14]:


#Logistic regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[15]:


model.fit(x_train,y_train)


# In[16]:


#print metric to get performance
print('Accuracy: ',model.score(x_train,y_train)*100)


# In[17]:


#KNN
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()


# In[18]:


model.fit(x_train,y_train)


# In[20]:


print('Accuracy :',model.score(x_test,y_test)*100)


# In[ ]:




