#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import numpy as np
import xlrd
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn import datasets
import calendar

import json
from datetime import date
from pandas_datareader import data

today = date.today()
start_date = "2010-01-01"
end_date = "{}".format(today)
df = data.DataReader("^N225", "yahoo", start_date, end_date)
#panel_data.to_csv(path_or_buf = location, index=False)


# In[2]:


df


# In[3]:


print(df.reset_index(level=0, inplace=True)) #it worked!


# In[4]:


df["Date"]


# In[5]:


date = df["Date"]


# In[6]:


price = df["Adj Close"]


# In[7]:


X = date


# In[8]:


y = price


# In[9]:


X = X.rename_axis("Date")


# In[10]:


y = y.rename_axis("Price")


# In[11]:


X = X.values.astype("datetime64[D]").astype(int)


# In[12]:


X = pd.Series(X)


# In[13]:


type(y)


# In[14]:


type(X)


# In[15]:


X = X.values.reshape(-1,1)


# In[16]:


Y = y.values.reshape(-1,1)


# In[ ]:





# In[17]:


lm = linear_model.LinearRegression()
model = lm.fit(X,y)


# In[18]:


predictions = lm.predict(X)
print(predictions)


# In[19]:


lm.score(X,y)


# In[20]:


lm.coef_


# In[21]:


lm.intercept_


# In[22]:


linear_regressor = LinearRegression()


# In[23]:


linear_regressor.fit(X, y)


# In[24]:


Y_pred = linear_regressor.predict(X)
Y_pred


# In[25]:


""""plt.plot(X, Y_pred, y, color="red", )
plt.show()"""


# In[26]:


plt.plot(X,y)
plt.plot(X, Y_pred, color="red")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




