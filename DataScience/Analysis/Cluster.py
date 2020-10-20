from sklearn.cluster import KMeans
from sklearn import cluster, datasets, mixture
import numpy as np


class Cluster(object):
    """description of class"""
    @staticmethod
    def kmeans(dataset, k):
        return KMeans(n_cluster=k).fit(dataset)

    @staticmethod
    def fit_to_center(dataset, center, axes):
        return KMeans(n_clusters=len(center),max_iter=0,init=center).fit(dataset[axes])

    @staticmethod
    def make_circles(n_samples, factor=.5, noise=.05):
        return datasets.make_circles(n_samples=n_samples, factor=factor, noise=noise)

    @staticmethod
    def make_moons(n_samples, noise=.05):
        return datasets.make_moons(n_samples=n_samples,  noise=noise)
    @staticmethod
    def make_blobs(n_samples, random_state=8):
        return datasets.make_blobs(n_samples=n_samples, random_state=8)



