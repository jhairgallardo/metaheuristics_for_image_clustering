import os
import cv2
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

import metrics
import utils
from metaheuristic_algos import metaheuristic
from k_means import kmeans
from metrics import *

SEED=128
np.random.seed(SEED)

def main(img_name,num_clusters,algorithm):

    ## Get image
    img = cv2.imread('inputs/{}.jpg'.format(img_name))

    ## Get datapoints from image
    datapoints = utils.getImageFeatures(img, 'RGB')

    ## Run clustering metho
    if algorithm['name']=='kmeans':  # Run kmeans
        labels, final_centroids = kmeans(datapoints,
                                     num_clusters=num_clusters)
    else:  # Run metaheuristic
        labels, final_centroids = metaheuristic(datapoints, algorithm, num_clusters)

    ## Create mask from assigned clusters
    mask = labels.reshape(img.shape[:2])
    mask = mask.astype(np.int)

    img_seg, cluster_centroids = utils.colorSegmentedImage(img.astype(np.float32) / 255, mask)

    ## Run metrics to measure
    ## quality of segmentation
    psnr_value  = metrics.PSNR(img/255, img_seg)
    inter_value = metrics.inter_cluster_distance(cluster_centroids)
    intra_value = metrics.intra_cluster_distance(img/255, mask, cluster_centroids)

    return psnr_value, inter_value, intra_value, img_seg


if __name__ == '__main__':
    ## Input image and number of clusters
    input_dict = {0:['lena', 6], 1:['tiger',8], 2:['cameraman', 4], 3:['coins', 2]} # .jpg images
    img_idx = 1 # (Change this line to use different images)
    img_name = input_dict[img_idx][0] # lena, cameraman, coins
    num_clusters = input_dict[img_idx][1] # 6 for lena, 4 for cameraman, 2 for coins 
    
    # PSO = Particle Swarm Optimization
    # DE = Differential Evolution
    # SA = Simulated Annealing
    # FA = Firefly Algorithm
    # BA = Bat Algorithm

    ## Algorithms parameters
    all_algos = {'kmeans': {'name':'kmeans', 'clusters': num_clusters},
                 'PSO'   : {'name':'PSO', 'alpha':1.5, 'beta':0.5, 'pop_size':20, 'iter':50, 'clusters': num_clusters},
                 'DE'    : {'name':'DE', 'F':1.0, 'Cr':0.5, 'pop_size':20, 'iter':50, 'clusters': num_clusters},
                 'SA'    : {'name':'SA', 'init_temp':10, 'beta':0.8, 'iter':1000, 'clusters': num_clusters}, # 1000 iters to be fair against pop based algos (20x50)
                 'FA'    : {'name':'FA', 'gamma':1, 'alpha':0.1, 'beta':1, 'pop_size':20, 'iter':50, 'clusters': num_clusters},
                 'BA'    : {'name':'BA', 'alpha':0.8, 'gamma':0.9, 'pop_size':20, 'iter':50, 'clusters': num_clusters}
                }

    ## Algorithm to use for image segmentation
    algorithm = all_algos['FA'] # (Change this line to use different algorithms)

    ## Set folder to save results
    savedir = f"results/{algorithm['name']}/{img_name}/"
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
        print('created folder: ', savedir)
     
    ## Run trials
    N = 5 # number of trials
    psnr_trials=[]
    inter_trials=[]
    intra_trials=[]
    bestpsnr=0
    for i in range(N):
        print('Running trial: ', i)
        psnr, inter, intra, img_seg = main(img_name, num_clusters, algorithm)
        psnr_trials.append(psnr)
        inter_trials.append(inter)
        intra_trials.append(intra)
        if psnr > bestpsnr:
            best_img_seg = deepcopy(img_seg)

    # Get statistics over trials
    psnr_mean = np.mean(psnr_trials)
    psnr_std = np.std(psnr_trials)
    inter_mean = np.mean(inter_trials)
    inter_std = np.std(inter_trials)
    intra_mean = np.mean(intra_trials)
    intra_std = np.std(intra_trials)

    # Print and save results
    lines = [img_name,
             algorithm['name'] + ' ' + str(num_clusters) + ' clusters',
             f'Number of trials {N}',
             f'PSNR: {psnr_mean} +- {psnr_std}',
             f"Inter-cluster distance: {inter_mean} +- {inter_std}",
             f"Intra-cluster distance: {intra_mean} +- {intra_std}"]
    with open(savedir+'metrics_{}_{}_clusters_{}.txt'.format(img_name,num_clusters,algorithm['name']),'w') as f:
        f.write('\n'.join(lines))
    
    for i in range(len(lines)):
        print(lines[i])

    # Save used algorithm parameters
    f = open(savedir+'parameters_{}_{}_clusters_{}.txt'.format(img_name,num_clusters,algorithm['name']),"w")
    f.write(str(algorithm))
    f.close()

    # Save best segmentation image across trials
    cv2.imwrite(savedir+'img_seg_{}_{}_clusters_{}.jpg'.format(img_name,num_clusters,algorithm['name']), best_img_seg*255)
