#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


path="E:\\Machine_learning_projects\\classification\\data.csv"
my_data=pd.read_csv(path,header=None,names=["Exam1","Exam2","Admitted"])


# In[5]:


my_data.head()


# In[6]:


my_data.describe()


# In[16]:


positive=my_data[my_data['Admitted'].isin([1])]
negative=my_data[my_data['Admitted'].isin([0])]
negative.head()


# In[15]:


positive.head()


# In[25]:


fig,ax=plt.subplots(figsize=(8,5))
ax.scatter(positive["Exam1"],positive["Exam2"],s=50,c='b',marker='o',label='Admitted')
ax.scatter(negative["Exam1"],negative["Exam2"],s=50,c='r',marker='x',label=' Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 score')
ax.set_ylabel('Exam 2 score')


# In[29]:


def sigmoid(z):
    return 1 / (1 + np.exp(-z))
nums = np.arange(-10, 10, step=1)
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(nums, sigmoid(nums), 'r')


# In[31]:


my_data.insert(0,'One',1)
my_data.head()


# In[32]:


cols=my_data.shape[1]
X=my_data.iloc[:,:cols-1]
y=my_data.iloc[:,cols-1:cols]


# In[53]:


X=np.array(X)
y=np.array(y)
theta=np.zeros(3)


# In[55]:


def cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))
thiscost=cost(theta,X,y)
thiscost


# In[62]:


#gradient
def gradient(theta,X,y):
    theta=np.matrix(theta)
    X=np.matrix(X)
    y=np.matrix(y)
    parameters=int(theta.ravel().shape[1])
    error=sigmoid(X*theta.T)-y
    grad=np.zeros(parameters)
    
    for i in range (parameters):
        term=np.multiply(error,X[:,i])
        grad[i]=np.sum(term)/len(X)
    return grad


# In[70]:


import scipy.optimize as opt
result=opt.fmin_tnc(func=cost,x0=theta,fprime=gradient,args=(X,y))
num_of_iteration =result[1]
num_of_iteration


# In[74]:


besttheta=result[0]
besttheta
costopt=cost(besttheta,X,y)
costopt


# In[83]:


def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]


# In[87]:


theta_min = np.matrix(result[0])
predictions = predict(theta_min, X)
correct = [1 if ((a == 1 and b == 1) or
                 (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
#predictions


# In[98]:


#accuracy
err=np.sum(correct-y)
acc=(sum(map(int,correct))%len(correct))
acc


# In[ ]:




