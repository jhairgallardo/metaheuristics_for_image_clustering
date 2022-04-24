import cv2
import numpy as np
import matplotlib.pyplot as plt

import metrics
import utils
from metaheuristic_algos import metaheuristic
from k_means import kmeans
from metrics import *


def main():
    ## Get image
    img_name = 'fruit'
    img = cv2.imread('inputs/{}.jpg'.format(img_name))

    ## Get datapoints from image
    datapoints = utils.getImageFeatures(img, 'RGB')

    ## TODO
    ## run metaheuristic algorithm
    ## to find initial centroids
    # Define cost_function
    # algorithm = 'PSO'
    # init_centroids = metaheuristic(datapoints, algorithm, 
    #                                cost_function,num_clusters)
    init_centroids = None

    ## Run kmeans
    num_clusters = 4
    labels, final_centroids = kmeans(datapoints,
                                     num_clusters=num_clusters,
                                     init_centroids=init_centroids)

    ## Create mask from assigned clusters
    mask = labels.reshape(img.shape[:2])
    mask = mask.astype(np.int)

    img_seg, cluster_centroids = utils.colorSegmentedImage(img.astype(np.float32) / 255, mask)

    ## Run metrics to measure
    ## quality of segmentation
    print(f"PSNR: {metrics.PSNR(img/255, img_seg)}")
    print(f"Inter-cluster distance: {metrics.inter_cluster_distance(cluster_centroids)}")
    print(f"Intra-cluster distance: {metrics.intra_cluster_distance(img/255, mask, cluster_centroids)}")

    # Plot segmented image
    plt.figure()
    plt.imshow(img_seg[:, :, ::-1])
    plt.title('{} clusters'.format(num_clusters))
    plt.axis('off')
    plt.savefig('results/{}_segmented.jpg'.format(img_name))
    plt.show()

    return None


if __name__ == '__main__':
    main()
