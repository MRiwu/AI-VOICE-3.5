import numpy as np
import torch
from sklearn.cluster import KMeans

checkpoint = torch.load("cluster/kmeans_500.pt")
kmeans_dict = {}
for spk, ckpt in checkpoint.items():
    km = KMeans(ckpt["n_features_in_"])
    km.__dict__["n_features_in_"] = ckpt["n_features_in_"]
    km.__dict__["_n_threads"] = ckpt["_n_threads"]
    km.__dict__["cluster_centers_"] = ckpt["cluster_centers_"]
    kmeans_dict[spk] = km

def get_cluster_result(x, speaker):
    """
        x: np.array [t, 256]
        return cluster class result
    """
    return kmeans_dict[speaker].predict(x)
def get_cluster_center_result(x,speaker):
    """x: np.array [t, 256]"""
    predict = kmeans_dict[speaker].predict(x)
    return kmeans_dict[speaker].cluster_centers_[predict]
def get_center(x,speaker):
    return kmeans_dict[speaker].cluster_centers_[x]

