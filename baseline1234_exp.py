# train_exploration_vlad.py
import os
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from natsort import natsorted
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree


def compute_all_sift(img_dir, sift):
    files = natsorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
    all_desc = []
    for fname in tqdm(files, desc="Extracting SIFT"):
        img = cv2.imread(os.path.join(img_dir, fname))
        if img is None:
            continue
        _, des = sift.detectAndCompute(img, None)
        if des is None:
            continue
        all_desc.append(des)
    if not all_desc:
        raise RuntimeError("No SIFT descriptors extracted, please check image folder.")
    all_desc = np.vstack(all_desc)
    return files, all_desc


def compute_vlad_for_img(img, sift, codebook):
    _, des = sift.detectAndCompute(img, None)
    if des is None or len(des) == 0:
        # no keypoints, return zero vector
        k = codebook.n_clusters
        d = codebook.cluster_centers_.shape[1]
        return np.zeros(k * d, dtype=np.float32)

    pred = codebook.predict(des)
    centers = codebook.cluster_centers_
    k = codebook.n_clusters
    vlad = np.zeros((k, des.shape[1]), dtype=np.float32)

    for i in range(k):
        mask = (pred == i)
        if np.any(mask):
            vlad[i] = np.sum(des[mask] - centers[i], axis=0)

    vlad = vlad.reshape(-1)
    # power-normalization
    vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))
    # L2
    norm = np.linalg.norm(vlad)
    if norm > 1e-12:
        vlad = vlad / norm
    return vlad


def main():
    IMG_DIR = "./data/images_subsample"
    ART_DIR = "./data/artfact1"   # 按你说的名字来
    os.makedirs(ART_DIR, exist_ok=True)

    sift = cv2.SIFT_create()

    # 1) 先把所有图的 SIFT 拿到
    files, all_desc = compute_all_sift(IMG_DIR, sift)

    # 2) 训 codebook
    # 可以自己改 n_clusters，如果你觉得 128 太大/太慢
    print("Training KMeans codebook...")
    kmeans = KMeans(
        n_clusters=128,
        init="k-means++",
        n_init=5,
        verbose=1,
        random_state=0
    ).fit(all_desc)

    # 3) 对每一张图做 VLAD
    vlad_db = []
    for fname in tqdm(files, desc="Building VLAD database"):
        img = cv2.imread(os.path.join(IMG_DIR, fname))
        v = compute_vlad_for_img(img, sift, kmeans)
        vlad_db.append(v)
    vlad_db = np.stack(vlad_db, axis=0)

    # 4) BallTree
    print("Building BallTree...")
    tree = BallTree(vlad_db, leaf_size=64)

    # 5) 存文件
    # 这些都是导航阶段会用到的
    np.save(os.path.join(ART_DIR, "vlad.npy"), vlad_db)
    np.save(os.path.join(ART_DIR, "idx2name.npy"), np.array(files))
    with open(os.path.join(ART_DIR, "codebook.pkl"), "wb") as f:
        pickle.dump(kmeans, f)
    with open(os.path.join(ART_DIR, "balltree.pkl"), "wb") as f:
        pickle.dump(tree, f)

    meta = {
        "image_dir": IMG_DIR,
        "n_images": len(files),
        "n_clusters": int(kmeans.n_clusters),
        "desc_dim": int(kmeans.cluster_centers_.shape[1]),
        "note": "VLAD + BallTree built from exploration data"
    }
    with open(os.path.join(ART_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Done. Artifacts saved to:", ART_DIR)


if __name__ == "__main__":
    main()
