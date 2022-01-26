#!/usr/bin/env python
# coding: utf-8

# In[8]:


"""Loading a data set from sklearn"""
import pandas as pd
from sklearn.datasets import load_iris
iris_data = load_iris()
iris_data


# In[23]:


df_x = pd.DataFrame(iris_data.data)
df_x

"""As the column name is not  given then we will manually give it"""

df_x.rename(columns = {0:"Sepal_length"}, inplace = True)
df_x.rename(columns = {1:"Sepal_width"}, inplace = True)
df_x.rename(columns = {2:"Petal_length"}, inplace = True)
df_x.rename(columns = {3:"Petal_width"}, inplace = True)
df_x.head()


# In[32]:


df_y = pd.DataFrame(iris_data.target)
df_y.rename(columns= {0:"Target_name"}, inplace = True)
df_y.head()


# In[34]:


"""Now concatinating df_x and df_y"""

Iris_Data = pd.concat([df_x,df_y], axis = 1)
Iris_Data.head()


# In[39]:


"""Now for understanding purpose we need to rename the Target names data points"""

for i in range(0, len(Iris_Data["Target_name"]), 1):
    if Iris_Data["Target_name"][i] == 0:
        Iris_Data["Target_name"][i] = "setosa"
    elif Iris_Data["Target_name"][i] == 1:
        Iris_Data["Target_name"][i] = "versicolor" 
    elif Iris_Data["Target_name"][i] == 2:
        Iris_Data["Target_name"][i] = "virginica"
        
        
Iris_Data.info()


# In[41]:


"""Train test splitting of the data"""

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df_x,df_y, test_size = 0.33, random_state = 4)


# In[45]:


print(X_train.count())
print(X_test.count())
print(Y_train.count())
print(Y_test.count())


# In[46]:


X_train.head()


# In[58]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,Y_train)
print(gnb.predict(X_test)) # I don't understand how this X_test became integer value.  It should be features
print(gnb.predict(Y_test))


# In[ ]:




