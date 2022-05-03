# Image Segmentation with Metaheuristic algorithms

## Abstract

Partitioning an image into regions based on a defined criteria is the task known as image segmentation. This can be done by clustering similar pixels into groups. A strong clustering-based method to segment images is k-means, which iteratively updates the centroids for each cluster starting from a random initialization. However, k-means final centroids highly depend on this random initialization and can easily fall into local optima solutions. On the other hand, metaheuristic methods are known for their robustness against random initialization when finding the global optima (or close to it) of a cost function, overcoming the k-means issue. While there are works on image segmentation using metaheuristic algorithms, a review of these algorithms in the same conditions and same datasets is missing. Here, we test several metaheuristic algorithms on the image segmentation task and compare them against k-means. We measure their performance using PSNR, inter-cluster, and intra-cluster distance on classic images used by the image processing community. We found that metaheuristic algorithms are able to perform image segmentation and get comparable results to k-means.