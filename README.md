# Image Segmentation with Metaheuristic algorithms

## Abstract

Partitioning an image into regions based on a defined criteria is the task known as image segmentation. This can be done by clustering similar pixels into groups. A strong clustering-based method to segment images is k-means, which iteratively updates the centroids for each cluster starting from a random initialization. However, k-means final centroids highly depend on this random initialization and can easily fall into local optima solutions. On the other hand, metaheuristic methods are known for their robustness against random initialization when finding the global optima (or close to it) of a cost function, overcoming the k-means issue. While there are works on image segmentation using metaheuristic algorithms, a review of these algorithms in the same conditions and same datasets is missing. Here, we test several metaheuristic algorithms on the image segmentation task and compare them against k-means. We measure their performance using PSNR, inter-cluster, and intra-cluster distance on classic images used by the image processing community. We found that metaheuristic algorithms are able to perform image segmentation and get comparable results to k-means.

## Requirements

This code was tested on `python 3.8.5`. It requires the following libraries:

```
numpy 1.19.2
cv2 4.5.5
scipy 1.6.0
```

The libraries `numpy` and `scipy` are included in a freash installation of python using [conda](https://www.anaconda.com/).

The [OpenCV](https://opencv.org/) library `cv2` can be installed as follows:

```
pip install opencv-python
```

It requires the `input` folder with the images to be segmented.


## Usage

To use this code, you only need to run the file `main.py` as follows:

```
python main.py
```

On file `main.py`, the variable `img_idx` on line 49 selects the image to be used. On line 69, changing the input argument to the dictionary `all_algos` defines the algorithm to be used.