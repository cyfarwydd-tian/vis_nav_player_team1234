# baseline1234_exp.py
# 离线建图脚本：支持“纹理 → 拟FPV视角”增强、RootSIFT、可选PCA白化
# 依赖：opencv-python, numpy, scikit-learn, natsort, tqdm

import os
import json
import pickle
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from natsort import natsorted
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors, BallTree
from sklearn.decomposition import PCA

# ============================ 全局参数（改这里就行） ============================
CONFIG = dict(
    OUT_DIR="./artifacts_baseline1234",  # 工件输出目录
    KMEANS_KMAX=64,                      # KMeans 最大簇数（VLAD维度 = 簇数*128）
    KNN_K=10,                            # kNN 图的 k
    # 如果数据是“纹理库”，设置 >0：每张纹理生成 N 个“拟FPV视角”
    # 若 data_dir 本身就是探索阶段的 FPV 帧，则设为 0（关闭增强）
    TEXTURE_AUG=8,
    AUG_OUT_SIZE=480,                    # 拟视角尺寸（方形）
    PREVIEW_DIR="./aug_preview",         # 增强预览目录；设为 "" 则不保存预览
    MIN_CANVAS=1024,                     # 小纹理平铺到的最小画布边长
    # 随机裁剪（针对“墙面取纹理小块”的迷宫）
    CROP_PROB=0.8,                       # 做随机裁剪的概率
    CROP_SCALE_MIN=0.25,                 # 裁剪后面积比例下界（0~1）
    CROP_SCALE_MAX=0.75,                 # 裁剪后面积比例上界（0~1）
    # SIFT 设置（RootSIFT 在描述子层做 L1+sqrt）
    SIFT_NFEATURES=800,
    SIFT_CONTRAST=0.02,
    SIFT_EDGE=10,
    # 可选：对 VLAD 做 PCA-白化（降维 + 提升区分度）
    USE_PCA_WHITEN=True,
    PCA_DIM=1024,                        # 目标维度（例如 64*128=8192 降到 1024）
    # 其他
    AUG_PREVIEW_MAX=24,
    RANDOM_STATE=0,
)
# ============================================================================

cv2.setNumThreads(0)  # 避免OpenCV抢太多线程；需要时改为1或删除


# ------------------------- 基础工具 -------------------------
def find_images_recursive(base_dir: str):
    base = Path(base_dir).resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"Image folder not found: {base}")
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    files = []
    for p in base.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(str(p.relative_to(base)))
    files = natsorted(files)
    if len(files) == 0:
        raise FileNotFoundError(f"No images found under {base}. Supported: {sorted(exts)}")
    return base, files


def train_codebook(des_pool: np.ndarray, k_max: int = 64):
    n_desc = int(des_pool.shape[0])
    k = min(k_max, max(8, n_desc // 20))  # 约5%描述子，至少8
    print(f"[KMeans] training with n_clusters={k} on {n_desc} descriptors ...")
    km = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=5,
        verbose=1,
        random_state=CONFIG["RANDOM_STATE"],
    ).fit(des_pool)
    return km


# ------------------------- RootSIFT + VLAD -------------------------
def _extract_rootsift(img: np.ndarray, sift: cv2.SIFT):
    kps, des = sift.detectAndCompute(img, None)
    if des is None or len(des) == 0:
        return None, None
    des = des.astype(np.float32)
    des /= (np.linalg.norm(des, ord=1, axis=1, keepdims=True) + 1e-12)  # L1
    des = np.sqrt(des)                                                   # sqrt
    return kps, des


def vlad_encode(img: np.ndarray, sift: cv2.SIFT, codebook) -> np.ndarray:
    _, des = _extract_rootsift(img, sift)
    k = codebook.n_clusters
    if des is None or len(des) == 0:
        return np.zeros(k * 128, dtype=np.float32)

    pred = codebook.predict(des)
    C = codebook.cluster_centers_.astype(np.float32)
    vlad = np.zeros((k, des.shape[1]), dtype=np.float32)

    # 残差聚合 + intra-normalization
    for i in range(k):
        mask = (pred == i)
        if np.any(mask):
            residual = des[mask].astype(np.float32) - C[i]
            v = residual.sum(axis=0)
            n = np.linalg.norm(v) + 1e-12
            vlad[i] = v / n

    vlad = vlad.reshape(-1)
    # power-law + L2
    vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))
    vlad /= (np.linalg.norm(vlad) + 1e-12)
    return vlad.astype(np.float32)


def build_knn_graph(X: np.ndarray, k: int = 10):
    k = min(k, len(X))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X)
    dist, ind = nn.kneighbors(X, return_distance=True)
    return ind.astype(np.int32), dist.astype(np.float32)


def save_artifacts(out_dir, meta, idx2name, codebook, vlads, knn_ind, knn_dist, balltree, pca_model=None):
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    np.save(out / "idx2name.npy", np.array(idx2name, dtype=object))
    with open(out / "codebook.pkl", "wb") as f:
        pickle.dump(codebook, f)

    np.save(out / "vlad.npy", vlads)
    np.save(out / "knn_ind.npy", knn_ind)
    np.save(out / "knn_dist.npy", knn_dist)

    try:
        with open(out / "balltree.pkl", "wb") as f:
            pickle.dump(balltree, f)
    except Exception as e:
        print(f"[warn] BallTree not pickled ({e}); will rebuild at navigation.)")

    if pca_model is not None:
        with open(out / "pca.pkl", "wb") as f:
            pickle.dump(pca_model, f)


# ------------------------- 纹理 → 拟FPV 视角增强 -------------------------
def _ensure_min_canvas(tex: np.ndarray, min_side=1024) -> np.ndarray:
    """把小纹理平铺到至少 min_side 的方形画布"""
    h, w = tex.shape[:2]
    rep_h = int(np.ceil(min_side / max(h, 1)))
    rep_w = int(np.ceil(min_side / max(w, 1)))
    big = np.tile(tex, (rep_h, rep_w, 1))
    return big[:min_side, :min_side]


def _random_perspective_warp(img: np.ndarray, out_size=480) -> np.ndarray:
    """随机透视：模拟不同俯仰/偏航看一块墙/地面"""
    H = W = out_size
    src = np.float32([[0, 0], [img.shape[1]-1, 0], [img.shape[1]-1, img.shape[0]-1], [0, img.shape[0]-1]])
    m = int(0.15 * out_size)  # 最大边缘内缩
    jitter = lambda: np.random.randint(0, m)
    dst = np.float32([
        [jitter(), jitter()],
        [W-1-jitter(), jitter()],
        [W-1-jitter(), H-1-jitter()],
        [jitter(), H-1-jitter()],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    return warped


def _photometric_jitter(img: np.ndarray) -> np.ndarray:
    """亮度/对比度/轻微模糊"""
    img = img.astype(np.float32)
    alpha = np.random.uniform(0.8, 1.2)  # 对比度
    beta = np.random.uniform(-15, 15)    # 亮度
    img = np.clip(img * alpha + beta, 0, 255).astype(np.uint8)
    if np.random.rand() < 0.3:
        k = np.random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    return img


def _random_crop(img: np.ndarray, scale_min=0.25, scale_max=0.75) -> np.ndarray:
    """随机裁剪（面积比例在 [scale_min, scale_max]）"""
    h, w = img.shape[:2]
    s = np.sqrt(np.random.uniform(scale_min, scale_max))  # 面积比例的平方根 → 边长比例
    ch, cw = int(h * s), int(w * s)
    if ch < 32 or cw < 32:
        return img
    y0 = np.random.randint(0, h - ch + 1)
    x0 = np.random.randint(0, w - cw + 1)
    return img[y0:y0 + ch, x0:x0 + cw]


def augment_texture_to_views(
    tex_bgr: np.ndarray,
    n_views=8,
    out_size=480,
    min_canvas=1024,
    crop_prob=0.8,
    crop_scale_min=0.25,
    crop_scale_max=0.75,
):
    """
    把一张纯纹理，转成 n_views 张“拟FPV视角”的切片：
    平铺→(可选)随机裁剪→透视→光照抖动
    """
    big = _ensure_min_canvas(tex_bgr, min_side=max(min_canvas, out_size * 2))
    views = []
    for _ in range(n_views):
        img = big
        if np.random.rand() < crop_prob:
            img = _random_crop(img, crop_scale_min, crop_scale_max)
        v = _random_perspective_warp(img, out_size=out_size)
        v = _photometric_jitter(v)
        views.append(v)
    return views


# ------------------------- 构建（支持两种模式） -------------------------
def compute_sift_pool(img_root, rel_paths, sift, texture_aug=0, out_size=480, preview_dir=None, min_canvas=1024,
                      crop_prob=0.8, crop_scale_min=0.25, crop_scale_max=0.75):
    pool = []
    valid_count, miss = 0, 0
    pv_count = 0

    for rel in tqdm(rel_paths, desc="SIFT detect (+aug)" if texture_aug > 0 else "SIFT detect"):
        img = cv2.imread(os.path.join(img_root, rel))
        if img is None:
            miss += 1
            continue

        if texture_aug > 0:
            views = augment_texture_to_views(
                img,
                n_views=texture_aug,
                out_size=out_size,
                min_canvas=min_canvas,
                crop_prob=crop_prob,
                crop_scale_min=crop_scale_min,
                crop_scale_max=crop_scale_max,
            )
            for v in views:
                _, des = _extract_rootsift(v, sift)
                if des is not None and len(des) > 0:
                    pool.append(des)
                    valid_count += 1
                if preview_dir and pv_count < CONFIG["AUG_PREVIEW_MAX"]:
                    outp = Path(preview_dir) / f"aug_pool_{pv_count:03d}.jpg"
                    outp.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(outp), v)
                    pv_count += 1
        else:
            _, des = _extract_rootsift(img, sift)
            if des is not None and len(des) > 0:
                pool.append(des)
                valid_count += 1

    if len(pool) == 0:
        raise RuntimeError("No SIFT descriptors detected from any image (after augmentation).")
    pool = np.vstack(pool).astype(np.float32)
    print(f"[SIFT] valid images(views): {valid_count}, failed to read: {miss}, descriptors: {pool.shape}")
    return pool


def build_vlad_database(img_root, rel_paths, sift, codebook, texture_aug=0, out_size=480, preview_dir=None,
                        min_canvas=1024, crop_prob=0.8, crop_scale_min=0.25, crop_scale_max=0.75):
    vlads = []
    names = []
    ok, miss = 0, 0
    pv_count = 0

    for rel in tqdm(rel_paths, desc="VLAD encode (+aug)" if texture_aug > 0 else "VLAD encode"):
        img = cv2.imread(os.path.join(img_root, rel))
        if img is None:
            miss += 1
            continue

        if texture_aug > 0:
            views = augment_texture_to_views(
                img,
                n_views=texture_aug,
                out_size=out_size,
                min_canvas=min_canvas,
                crop_prob=crop_prob,
                crop_scale_min=crop_scale_min,
                crop_scale_max=crop_scale_max,
            )
            for i, v in enumerate(views):
                vec = vlad_encode(v, sift, codebook)
                vlads.append(vec)
                names.append(f"{rel}::aug{i}")
                ok += 1
                if preview_dir and pv_count < CONFIG["AUG_PREVIEW_MAX"]:
                    outp = Path(preview_dir) / f"aug_vlad_{pv_count:03d}.jpg"
                    outp.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(outp), v)
                    pv_count += 1
        else:
            vec = vlad_encode(img, sift, codebook)
            vlads.append(vec)
            names.append(rel)
            ok += 1

    if len(vlads) == 0:
        raise RuntimeError("VLAD database empty: all images failed to load or encode.")
    vlads = np.vstack(vlads).astype(np.float32)
    print(f"[VLAD] encoded={ok}, failed_read={miss}, shape={vlads.shape}")
    return vlads, names


# ------------------------- 主流程 -------------------------
def main():
    parser = argparse.ArgumentParser(description="Offline map builder (texture→FPV augmentation, RootSIFT, optional PCA)")
    parser.add_argument("data_dir", nargs="?", default=None, help="Path to images root. If textures, enhancement per CONFIG.")
    args = parser.parse_args()

    if args.data_dir is None:
        raise SystemExit("Usage: python baseline1234_exp.py <path-to-images-or-textures>")

    img_root, files = find_images_recursive(args.data_dir)
    print(f"[scan] {len(files)} images under {img_root}")
    if CONFIG["TEXTURE_AUG"] > 0:
        print(f"[aug] texture→FPV: {CONFIG['TEXTURE_AUG']} views/image, out_size={CONFIG['AUG_OUT_SIZE']}, crop_prob={CONFIG['CROP_PROB']}")

    # SIFT（探测），描述子在 _extract_rootsift 里进行 RootSIFT 处理
    sift = cv2.SIFT_create(
        nfeatures=CONFIG["SIFT_NFEATURES"],
        contrastThreshold=CONFIG["SIFT_CONTRAST"],
        edgeThreshold=CONFIG["SIFT_EDGE"],
    )

    # 1) 训练 codebook
    des_pool = compute_sift_pool(
        str(img_root), files, sift,
        texture_aug=CONFIG["TEXTURE_AUG"],
        out_size=CONFIG["AUG_OUT_SIZE"],
        preview_dir=(CONFIG["PREVIEW_DIR"] or None),
        min_canvas=CONFIG["MIN_CANVAS"],
        crop_prob=CONFIG["CROP_PROB"],
        crop_scale_min=CONFIG["CROP_SCALE_MIN"],
        crop_scale_max=CONFIG["CROP_SCALE_MAX"],
    )
    codebook = train_codebook(des_pool, k_max=CONFIG["KMEANS_KMAX"])

    # 2) 构建 VLAD 数据库
    vlads_raw, names = build_vlad_database(
        str(img_root), files, sift, codebook,
        texture_aug=CONFIG["TEXTURE_AUG"],
        out_size=CONFIG["AUG_OUT_SIZE"],
        preview_dir=(CONFIG["PREVIEW_DIR"] or None),
        min_canvas=CONFIG["MIN_CANVAS"],
        crop_prob=CONFIG["CROP_PROB"],
        crop_scale_min=CONFIG["CROP_SCALE_MIN"],
        crop_scale_max=CONFIG["CROP_SCALE_MAX"],
    )

    # 3) 可选 PCA 白化
    pca_model = None
    vlads_for_index = vlads_raw
    if CONFIG["USE_PCA_WHITEN"]:
        keep = min(CONFIG["PCA_DIM"], vlads_raw.shape[1])
        print(f"[PCA] whitening to {keep} dims ...")
        pca_model = PCA(n_components=keep, whiten=True, random_state=CONFIG["RANDOM_STATE"])
        vlads_for_index = pca_model.fit_transform(vlads_raw)

    # 4) 索引
    print("[index] building BallTree ...")
    balltree = BallTree(vlads_for_index, leaf_size=64)
    print("[index] building kNN graph ...")
    knn_ind, knn_dist = build_knn_graph(vlads_for_index, k=CONFIG["KNN_K"])

    # 5) 保存
    meta = {
        "image_root": str(img_root),
        "vlad_dim_raw": int(vlads_raw.shape[1]),
        "vlad_dim": int(vlads_for_index.shape[1]),
        "n_images": int(vlads_for_index.shape[0]),
        "knn_k": int(knn_ind.shape[1]),
        "texture_aug": int(CONFIG["TEXTURE_AUG"]),
        "aug_out_size": int(CONFIG["AUG_OUT_SIZE"]),
        "use_pca": bool(CONFIG["USE_PCA_WHITEN"]),
        "pca_dim": int(vlads_for_index.shape[1]) if CONFIG["USE_PCA_WHITEN"] else 0,
        "crop_prob": float(CONFIG["CROP_PROB"]),
        "crop_scale_min": float(CONFIG["CROP_SCALE_MIN"]),
        "crop_scale_max": float(CONFIG["CROP_SCALE_MAX"]),
    }
    save_artifacts(
        out_dir=CONFIG["OUT_DIR"],
        meta=meta,
        idx2name=names,
        codebook=codebook,
        vlads=vlads_for_index,   # 注意：保存的是“用于索引的向量”（已PCA或未PCA）
        knn_ind=knn_ind,
        knn_dist=knn_dist,
        balltree=balltree,
        pca_model=pca_model,
    )
    print(f"[done] artifacts saved to: {Path(CONFIG['OUT_DIR']).resolve()}")


if __name__ == "__main__":
    main()
