#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[10]:


data=pd.read_csv("E:\TUSHAR\ASSINMENT DONE BY TUSHAR\ASSIGNMENT NO 7 (CLUSTERING)\EastWestAirlines1.csv")


# In[11]:


data.shape


# In[12]:


data.shape


# In[13]:


data.head()


# In[14]:


data.isna().sum()


# In[15]:


data.info()


# In[16]:


data.head(2)


# In[17]:


def norm_fun(i):
    x=((i-i.min())/(i.max()-i.min()))
    return(x)


# In[18]:


df_norm=norm_fun(data.iloc[:,1:])
df_norm


# In[21]:


dendrogram=sch.dendrogram(sch.linkage(df_norm,method='complete'))


# In[22]:


hc=AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage='complete')


# In[23]:


hc


# In[24]:


y_hc=hc.fit_predict(df_norm)


# In[25]:


cluster=pd.DataFrame(y_hc,columns=['clusters'])
cluster.value_counts()


# In[26]:


from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


# In[27]:


df_norm.head()


# In[28]:


model=KMeans()
model.fit(df_norm)
model.labels_


# In[30]:


md=pd.Series(model.labels_)
data['clust']=md
df_norm.head()


# In[32]:


data.groupby(data.clust).mean()


# In[31]:


data.head()


# In[34]:


from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


# In[35]:


array=data.values
array


# In[36]:


stscaler=StandardScaler().fit(array)
x=stscaler.transform(array)


# In[37]:


dbscan=DBSCAN(eps=0.8,min_samples=6)
dbscan.fit(x)
dbscan.labels_


# In[38]:


cl=pd.DataFrame(dbscan.labels_,columns=['Cluster'])
cl

