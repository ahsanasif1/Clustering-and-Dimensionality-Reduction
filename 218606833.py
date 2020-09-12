#!/usr/bin/env python
# coding: utf-8

# # MUHAMMAD AHSAN ASIF
# # 218606833
#  Kindly read the pdf for Question 1, part 3 and 4 explanation.

# In[77]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from nltk.cluster import KMeansClusterer
import nltk
from sklearn import cluster
from sklearn import metrics
from sklearn.preprocessing import scale


# # Part 1 - Clustering

# 1)


#QUESTION 1

#Creating a dataframe M from the csv
M = pd.read_csv('digitData3.csv')
print('Dimensions of M matrix:', M.shape)

# Assigning m rows and n-1 columns from M dataframe
X=M.values[:,:-1]
print('Dimensions of X matrix:', X.shape)

#Assigning nth column of M 
trueLabels=M.values[:,-1]
print('Dimensions of trueLabels:', trueLabels.shape)




#QUESTION 2

#Computing kmeans for 5 clustering
kmeans = KMeans(n_clusters=5)
kmeans.fit(X)

#Getting the centroids and labels
centroids = kmeans.cluster_centers_
labels    = kmeans.labels_



#computing adjusted rand index
print(adjusted_rand_score(trueLabels,labels))


#computing adjusted mutual information 
print(adjusted_mutual_info_score(trueLabels,labels,'arithmetic'))



averaged_rand_score = 0
averaged_mutual_score = 0
i=0


while i<50:
    #running the kmeans for 50 random initialization
    kmeanstest = KMeans(n_clusters=5,init='random',n_init=1)
    kmeanstest.fit(X)
    centroids = kmeans.cluster_centers_
    labels    = kmeans.labels_
    
    #Calculating the average of adjusted rand scord and adjusted mutual information
    ad_rand = adjusted_rand_score(trueLabels,labels)
    ad_mutual = adjusted_mutual_info_score(trueLabels,labels,'arithmetic')
    
    
    averaged_rand_score = averaged_rand_score+ad_rand
    averaged_mutual_score = averaged_mutual_score + ad_mutual
    i += 1



#adjusted mutual information score after 50 iterations
averaged_mutual_score = averaged_mutual_score/50
print('The average adjusted mutual information score after 50 iterations : ', averaged_mutual_score)


#adjusted rand score after 50 iterations
averaged_rand_score = averaged_rand_score/50
print('The average adjusted rand score score after 50 iterations : ', averaged_rand_score)


# 3)



#Now for instance we had an initial value of 0.7 as ARI

ARI = 0.7
k=0
# Lets run the kmeans for 20 runs 

while k<20:
    #running the kmeans for 20 kmeans++ initialization
    kmeanstest = KMeans(n_clusters=5,init='k-means++',n_init=1)
    kmeanstest.fit(X)
    centroids3 = kmeans.cluster_centers_
    labels3    = kmeans.labels_
    
    #Calculating the average of adjusted rand scord 
    ad_rand3 = adjusted_rand_score(trueLabels,labels3)
    ARI = ARI+ad_rand3
    k +=1




#After running for 20 runs we can see that the ARI is as followed
ARI = ARI/20
print(ARI)


#After seeing the value of ARI, it has been decreased from 0.7 to 0.39. It could be concluded that with every single run the accuracy between the similarity of predicted labels and trained labels were decreasing. When computing kmeans ++ the value should be nearer to 1 otherwise if close to 0 it would indicate that centroids were initialized at random which is not the case. The higher the value of ARI the more similarity there is between the predicted and trained labels, as we have seen. Also, the less value of ARI could also be due to outliers, which might be chosen as data points. Kmeans++ also always selects the centroids which are far away from the data points as that increases its efficient and accuracy in overall results.


# 4)



# Doing kmeans through cosine distance similarity measure from the NLTK library
# Initializing the clusters to use cosine distance

kclusterer = KMeansClusterer(5, distance=nltk.cluster.util.cosine_distance, repeats=25)
assigned_clusters = kclusterer.cluster(X, assign_clusters=True)

kmeans_cos = cluster.KMeans(n_clusters=5)
kmeans_cos.fit(X)

cosine_centroids = kmeans_cos.cluster_centers_
cosine_labels = kmeans_cos.labels_




#computing adjusted rand index
print(adjusted_rand_score(trueLabels,cosine_labels))


# In[15]:


#computing adjusted mutual information 
print(adjusted_mutual_info_score(trueLabels,cosine_labels,'arithmetic'))


# In[16]:


averaged_rand_score2 = 0
averaged_mutual_score2 = 0
j=0


# In[17]:


while j<50:
    #running the kmeans using cosine distance for 50 random initialization
    
    kclusterer = KMeansClusterer(5, distance=nltk.cluster.util.cosine_distance, repeats=25)
    assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
    kmeans_cos = cluster.KMeans(n_clusters=5)
    kmeans_cos.fit(X)

    cosine_centroids = kmeans_cos.cluster_centers_
    cosine_labels = kmeans_cos.labels_
    
    #Calculating the average of adjusted rand scord and adjusted mutual information
    ad_rand2 = adjusted_rand_score(trueLabels,cosine_labels)
    ad_mutual2 = adjusted_mutual_info_score(trueLabels,cosine_labels,'arithmetic')
    
    
    averaged_rand_score2 = averaged_rand_score2+ad_rand2
    averaged_mutual_score2 = averaged_mutual_score2 + ad_mutual2
    j += 1



#adjusted mutual information score after 50 iterations
averaged_mutual_score2 = averaged_mutual_score2/50
print('The average adjusted mutual information score after 50 iterations : ', averaged_mutual_score2)


#adjusted rand score after 50 iterations
averaged_rand_score2 = averaged_rand_score2/50
print('The average adjusted rand score score after 50 iterations : ', averaged_rand_score2)



#Before the comparison, just want to report that this method took me 25 minutes to run this loop for cosine distance. This method could be very costly for companies as it sacrifices the run time. Now if we compare the average ARI values from the kmeans++ using euclidean distance with random kmeans using cosine, we can see that the values are better than euclidean distance. This means that cosine distance gave more similarity between the predicted and trained data, which is good but at the cost of run time. It took 25 times the time euclidean distance took to compute the results. Therefore, i would conclude that cosine distance as a similarity measure to give more accurate results but at the cost of run time, if compared to kmeans with euclidean distance as a similarity measure.


# # Question 2

# # 1)



#Normalizing X
Xnorm = scale(X)


#Performing PCA for 40 dimensions
pca = PCA(n_components=40)


pca.fit(Xnorm)


#Calculating Variance
var= pca.explained_variance_ratio_


#Calculating cumulative variance
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)



# As we can see that if i use 40 dimensions i can get a variance of atleast 95% by then 
print(var1)




#Plotting the captured variance with respect to increasing latent dimensionalioty 
plt.plot(var1)
plt.xlabel("Principal components")
plt.ylabel("Variance captured")


# # 2)



#Performing PCA for two principle components

#Scaling the data
scaler = StandardScaler()
scaler.fit(X)

#Transforming the data
scaled_data = scaler.transform(X)

#Performing PCA for 2 principle components
pca2 = PCA(n_components= 2)
pca2.fit(scaled_data)
x_pca = pca2.transform(scaled_data)



#Now if we check the dimensions of x_pca we can see our two principle components as dimensions
print(x_pca.shape)


# Plotting the total rows of Xprojected onto the first two principal components
plt.figure(figsize=(8,6))
plot = plt.scatter(x_pca[:,0],x_pca[:,1],c = trueLabels, cmap='plasma')
plt.xlabel('First Principle Component, V1')
plt.ylabel('Second Principle Component, V2')
plt.title('Scatter plot of Rows and principle components')
plt.colorbar(plot)


#END




