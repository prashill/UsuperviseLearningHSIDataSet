#PRASHIL SONI (S00375453)
#HW4
#Machine learning #SPRING 2021

# In[18]:
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
# In[Reads the hyperspectral image]

pillsData = loadmat('pills.mat')['hsiImage']

# In[Sample code for printing a matrix with cluster values]

img = pillsData[:,:,136]
plt.matshow(img)
plt.xlabel("Sample")

pillsData1 = pillsData.reshape((pillsData.shape[0]*pillsData.shape[1]),pillsData.shape[2])

print("Printing pillslabel:",pillsData1)


# # K Means

# In[12]:


# Applied Kmeans
kmeans = KMeans(n_clusters=3)

# Train K-Means.
label = kmeans.fit_predict(pillsData1)

# print(kmeans)
# Evaluate the K-Means clustering accuracy.
# metrics.acc(y, y_pred_kmeans)
# print(k_mean.cluster_centers_)

let2d = np.reshape(label,(-1,150))

# print(let2d.shape)
# print(let2d[:][:])

#printing result using matplotlibrary
plt.matshow(let2d)
plt.xlabel("KMeans")


# # Agglomerative Clustering

# In[3]:


from sklearn.cluster import AgglomerativeClustering
# applied AgglomerativeClustering.
ac = AgglomerativeClustering(n_clusters =3, affinity='euclidean', linkage='ward')

# train AgglomerativeClustering.
label = ac.fit_predict(pillsData1)

#print("printing label of AgglomerativeClustering:",label)

let2d = np.reshape(label,(-1,150))

#print(let2d.shape)
# print(let2d[:][:])

#printing result using matplotlibrary
plt.matshow(let2d)
plt.xlabel("Agglomerative Clustering")


# # K Means With PCA

# In[5]:


from sklearn.decomposition import PCA

#applied PCA
pca = PCA(5)
 
#Transform the data
df = pca.fit_transform(pillsData1)
 
#print(df.shape)


# In[6]:

# applied Kmeans    
kmeans = KMeans(n_clusters=3)

# Train K-Means with PCA.
label = kmeans.fit_predict(df)

# print(k_mean)
# Evaluate the K-Means clustering accuracy.
# metrics.acc(y, y_pred_kmeans)
# print(k_mean.cluster_centers_)

let2d = np.reshape(label,(-1,150))
# print(let.shape)
# print(let[:][:])

#printing result using matplotlibrary 
plt.matshow(let2d)
plt.xlabel("K-Means with PCA")


# # Agglomerative Clustering with PCA

# In[8]:
from sklearn.decomposition import PCA

pca = PCA(2)
 
#Transform the data
df = pca.fit_transform(pillsData1)
 
print(df.shape)


# In[11]:
from sklearn.cluster import AgglomerativeClustering

# Train AgglomerativeClustering.
ac = AgglomerativeClustering(n_clusters =3, affinity='euclidean', linkage='ward')

# Train AgglomerativeClustering with PCA.
label = ac.fit_predict(df)
# print(label)

let2d = np.reshape(label,(-1,150))
# print(let2d.shape)
# print(let2d[:][:])

#printing result using matplotlibrary
plt.matshow(let2d)
plt.xlabel("Agglomerative Clustering with PCA")

