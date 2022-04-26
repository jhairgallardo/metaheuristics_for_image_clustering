import utils
import random
import numpy as np
from copy import copy
import matplotlib.pyplot as plt


def kmeans(points,num_clusters,init_centroids=None):

    if init_centroids is not None:
        centroids = np.array(init_centroids)
    else:
        rand_indx = random.sample(range(0,points.shape[0]),num_clusters)
        centroids = points[rand_indx]
        
    it = 0    
    while True:
        it+=1
        # Assign cluster
        SAD = []
        for centroid in centroids:
            SAD.append(np.sum(np.abs(points-centroid),axis=1))
        labels = np.argmin(SAD,axis=0)

        # Calculate new centroids
        new_centroids = []
        for i in range(num_clusters):
            new_centroids.append(np.mean(points[labels==i],axis=0))
        new_centroids = np.array(new_centroids)
        
        #print('kmeans iterations '+str(it),end='\r')
        # Out condition
        if np.sum(np.abs(centroids-new_centroids))==0:
            break
            
        # replacing old centroids with new centroids
        centroids=copy(new_centroids)
        
        # If I get bad initial centroids (cluster without points)
        if np.isnan(np.sum(np.abs(centroids-new_centroids))):
            rand_indx = random.sample(range(0,points.shape[0]),num_clusters)
            centroids = points[rand_indx]
            it=0

    # Assign cluster
    SAD = []
    for centroid in centroids:
        SAD.append(np.sum(np.abs(points-centroid),axis=1))
    labels = np.argmin(SAD,axis=0)
    return labels, centroids


    