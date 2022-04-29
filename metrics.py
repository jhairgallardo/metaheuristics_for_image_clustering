import numpy as np
from scipy.spatial import distance_matrix


def PSNR(original_im, segmented_im, pixel_max=1):
    """
    Calculates the PSNR of two images. A higher PSNR generally
    indicates a better segmentation.
    :param original_im: original image
    :param segmented_im: segmented image
    :param pixel_max: maximum value of pixels
    :return: PSNR
    """
    mse = np.mean((original_im - segmented_im) ** 2)
    if mse == 0:
        # no difference between images
        return 100
    return 20 * np.log10(pixel_max / np.sqrt(mse))


def inter_cluster_distance(cluster_centers):
    """
    Calculates the minimum inter-cluster distance. A higher inter-cluster
    distance is better.
    :param cluster_centers: array of vectors containing cluster centers
    :return: inter-cluster distance
    """
    dists = distance_matrix(cluster_centers, cluster_centers)
    remove_self = dists[dists > 0]
    return np.amin(remove_self)


def intra_cluster_distance(original_im, labeled_im, cluster_centers):
    """
    Calculates the maximum intra-cluster distance. A lower intra-cluster
    distance is better.
    :param original_im: original image
    :param labeled_im: labeled image
    :param cluster_centers: array of vectors containing cluster centers
    :return: maximum intra-cluster distance
    """
    cluster_distances = [np.sum(distance_matrix(original_im[labeled_im == i],
                         np.expand_dims(cluster_centers[i], axis=0)))/len(labeled_im[labeled_im == i])
                         for i in range(len(cluster_centers))]
    return np.nanmax(cluster_distances)