from scipy.io import loadmat
import matplotlib.pyplot as plt

# In[Reads the hyperspectral image]

pillsData = loadmat('pills.mat')['hsiImage']

# In[Sample code for printing a matrix with cluster values]
img = pillsData[:,:,10]
plt.matshow(img)

#In your program, each element of img matrix will be a cluster number. 
#So you can print the segmentation image on the screen. 
