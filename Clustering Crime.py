#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


# In[2]:


data=pd.read_csv("D:/ASSIGNMENTS/Clustering/crime_data.csv")
data.head()


# In[3]:


data.isna().sum()


# In[4]:


data[data.duplicated()]


# In[5]:


def norm_func(i):
    x=(i-i.min()/i.max()-i.min())
    return(x)


# In[6]:


df_norm=norm_func(data.iloc[:,1:])
df_norm


# In[7]:


dendrogram=sch.dendrogram(sch.linkage(df_norm,method='single'))


# In[8]:


hc=AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage='single')


# In[21]:


y_hc=hc.fit_predict(df_norm)
y_hc


# In[10]:


clusters=pd.DataFrame(y_hc,columns=['Clusters'])
clusters


# In[11]:


from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


# In[12]:


model=KMeans(n_clusters=4)
model.fit(df_norm)
model.labels_


# In[19]:


md=pd.Series(model.labels_)
data['clust']=md
df_norm.head()


# In[20]:


data.iloc[:,1:7].groupby(data.clust).mean()


# In[31]:


from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


# In[40]:


crime=pd.read_csv("D:/ASSIGNMENTS/Clustering/crime_data.csv")
crime.head()


# In[59]:


crime1=crime.iloc[:,1:6]
crime1.head()


# In[61]:


array=crime1.values
array


# In[63]:


stscaler = StandardScaler().fit(array)
X = stscaler.transform(array)


# In[64]:


dbscan=DBSCAN(eps=0.8,min_samples=5)


# In[65]:


dbscan.fit(X)


# In[66]:


dbscan.labels_


# In[68]:


cl=pd.DataFrame(dbscan.labels_,columns=["cluster"])
cl


# In[ ]:




