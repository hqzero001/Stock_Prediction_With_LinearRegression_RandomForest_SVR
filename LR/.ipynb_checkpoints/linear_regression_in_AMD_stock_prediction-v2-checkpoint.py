#!/usr/bin/env python
# coding: utf-8

# #Linear Regression in AMD stock price prediction

# In[64]:


#import libraries
import pandas as pd
import numpy as np
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[65]:


#load data
datafile = pd.read_csv("AMD (1980 -11.07.2023).csv")
datafile.head()


# In[67]:


#Add Prev_close column to store previous close price
series_shifted = datafile['Close'].shift()
datafile['Prev_close'] = series_shifted
datafile.head()


# In[69]:


#Drop unused column for train data
datafile = datafile.drop(columns = ['Adj Close'])
datafile.head()


# In[71]:


datafile.shape


# In[73]:


#null entries check (need to remove those data)
datafile.isnull().sum()


# In[75]:


#drop / remove NaN row or column
#inplace = true to execute dropna right on it file so python
#dont have to create a copy and we have to re-initialize and
#add to a new datafile variable
datafile.dropna(inplace = True)
datafile


# In[77]:


#check file info
datafile.info()


# In[79]:


datafile.describe()


# In[81]:


#plot close price (Draw close price figure)
datafile['Close'].plot(figsize = (10, 8))
plt.title("Close Price Overtime")
plt.xlabel("Index")
plt.ylabel("Close Price")
plt.show()


# In[82]:


#define target x and y (calculate close)
x = datafile[['Open', 'Prev_close', 'High', 'Low']]
y = datafile['Close']


# In[83]:


#allocate data for training
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0)


# In[84]:


print(x_train.shape, x_test.shape)


# In[85]:


#linear regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score
regressor = LinearRegression()
regressor.fit(x_train, y_train)


# In[86]:


#print regressor
#
print(regressor.coef_)
print(regressor.intercept_)


# In[87]:


#predicted value
predicted = regressor.predict(x_test)
print(x_test)
print(predicted)


# In[88]:


predicted.shape


# In[89]:


#Comparison predicted to actual test
_datafile = pd.DataFrame({"Actual": y_test, "Predicted": predicted})
print(_datafile)


# In[90]:


#Score
regressor.score(x_test, y_test)


# In[91]:


import math
print("Mean Absolute Error ", metrics.mean_absolute_error(y_test, predicted))
print("Mean Square Error ", metrics.mean_squared_error(y_test, predicted))
print("Root Mean Error ", math.sqrt(metrics.mean_squared_error(y_test, predicted)))


# In[92]:


predicted = regressor.predict(x)
print(predicted)


# In[93]:


#plot the graph
plt.plot(y, label = "Actual")
plt.plot(predicted, label = "Predicted")
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Actual vs Predicted")
plt.legend()
plt.show()


# In[102]:


#load test data
new_data = pd.read_csv("AMD (2023 - 08.04.2024).csv")
new_data.head()


# In[103]:


series_shifted = new_data['Close'].shift()
new_data['Prev_close'] = series_shifted
new_data.head()


# In[104]:


#Drop unused column for test data
new_data = new_data.drop(columns = ['Adj Close'])
new_data.head()


# In[105]:


new_data.shape


# In[106]:


#null entries check (need to remove those data)
new_data.isnull().sum()


# In[107]:


new_data.dropna(inplace = True)
new_data


# In[108]:


new_data.info()


# In[109]:


new_data.describe()


# In[110]:


#plot close price (Draw close price figure)
new_data['Close'].plot(figsize = (10, 8))
plt.title("Close Price Overtime")
plt.xlabel("Index")
plt.ylabel("Close Price")
plt.show()


# In[114]:


#define target x and y (calculate close)
x_new = new_data[['Open', 'Prev_close', 'High', 'Low']]
y_new = new_data['Close']


# In[115]:


#predicted value with new data
new_predicted = regressor.predict(x_new)
print(x_new)
print(new_predicted)


# In[116]:


new_predicted.shape


# In[117]:


#Comparison predicted to actual test
_new_data = pd.DataFrame({"Actual": y_new, "Predicted": new_predicted})
print(_new_data)


# In[119]:


#Score
regressor.score(x_new, y_new)


# In[122]:


import math
print("Mean Absolute Error ", metrics.mean_absolute_error(y_new, new_predicted))
print("Mean Square Error ", metrics.mean_squared_error(y_new, new_predicted))
print("Root Mean Error ", math.sqrt(metrics.mean_squared_error(y_new, new_predicted)))


# In[123]:


new_predicted = regressor.predict(x_new)
print(new_predicted)


# In[124]:


#plot the graph
plt.plot(y_new, label = "Actual")
plt.plot(new_predicted, label = "Predicted")
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Actual vs Predicted")
plt.legend()
plt.show()


# In[125]:


get_ipython().system('jupyter nbconvert --to script your_notebook_name.ipynb')


# In[ ]:




