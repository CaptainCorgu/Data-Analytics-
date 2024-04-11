#!/usr/bin/env python
# coding: utf-8

# In[5]:


pip install scikit-learn


# In[6]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# In[7]:


import pandas as pd
import os


# In[8]:


os.chdir ('C:\\Users\\alexi\\Downloads\\heart+failure+clinical+records')


# In[9]:


Heart = pd.read_csv(r'\Users\alexi\Downloads\heart+failure+clinical+records\heart_failure_clinical_records_dataset.csv') 


# In[10]:


Heart


# In[11]:


df = pd.DataFrame(Heart)


# In[12]:


#X andf Y defined 
x = df[['serum_sodium', 'serum_creatinine']]
y = df['platelets']


# In[13]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# In[14]:


#create the linear regression model 
model = LinearRegression()
model.fit(x_train, y_train)


# In[15]:


#can now be used for predictions 
y_pred = model.predict(x_test)


# In[16]:


#calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


# In[17]:


plt.scatter(x_test.iloc[:, 0], y_test, color='blue', label='Actual values')
plt.plot(x_test.iloc[:, 0], y_pred, color='red', label='Regression line')
plt.title('Linear Regression')
plt.legend()
plt.show()


# In[ ]:




