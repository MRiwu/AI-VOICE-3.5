import os
from pathlib import Path
import logging
import argparse

import torch
import numpy as np
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import time

def cluster(args):

    logger.info(f"Loading features from {args.in_dir}")
    features = []
    nums = 0
    for path in os.listdir(args.in_dir):
        if path.endswith("npy"):
            nums+=1
            features.append(np.load(os.path.join(args.in_dir, path)))
    features = np.concatenate(features, axis=0)
    print(nums, features.nbytes/ 1024**2, "MB , shape:",features.shape)

    logger.info(f"Clustering features of shape: {features.shape}")
    t = time.time()
    kmeans = KMeans(n_clusters=args.n_clusters,verbose=True).fit(features)
    print(time.time()-t, "s")
    checkpoint_path = args.checkpoint_dir / f"kmeans_{args.n_clusters}.pt"
    checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
    x = {
            "n_features_in_": kmeans.n_features_in_,
            "_n_threads": kmeans._n_threads,
            "cluster_centers_": kmeans.cluster_centers_,
    }
    print("end")
    torch.save(
        x,
        checkpoint_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster speech features features.")
    parser.add_argument(
        "in_dir",
        metavar="in-dir",
        help="path to the encoded dataset",
        type=Path,
    )
    # parser.add_argument(
    #     "subset",
    #     matavar="subset",
    #     help="path to the .txt file containing the list of files to cluster",
    #     type=Path,
    # )
    parser.add_argument(
        "checkpoint_dir",
        metavar="checkpoint-dir",
        help="path to the checkpoint directory",
        type=Path,
    )
    parser.add_argument(
        "--n-clusters",
        help="number of clusters",
        type=int,
        default=100,
    )
    args = parser.parse_args()
    cluster(args)
