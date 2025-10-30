# baseline1234_nav.py
# 纯自动导航玩家：不从 server 读目标，默认同目录下有 target.jpg
# 依赖：OpenCV、NumPy、scikit-learn（BallTree），以及 baseline1234_exp.py 生成的工件
from __future__ import annotations
import os
import json
import pickle
import math
import logging
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
from sklearn.neighbors import BallTree

from vis_nav_game import Player, Action, Phase

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def vlad_encode(img_bgr: np.ndarray, sift: cv2.SIFT, codebook) -> np.ndarray:
    """
    与离线阶段一致的 VLAD 编码：残差聚合 + intra-normalization + power-law + L2
    """
    _, des = sift.detectAndCompute(img_bgr, None)
    k = codebook.n_clusters
    if des is None or len(des) == 0:
        return np.zeros(k * 128, dtype=np.float32)

    pred = codebook.predict(des)
    C = codebook.cluster_centers_.astype(np.float32)
    vlad = np.zeros((k, des.shape[1]), dtype=np.float32)

    for i in range(k):
        mask = (pred == i)
        if np.any(mask):
            residual = des[mask].astype(np.float32) - C[i]
            v = residual.sum(axis=0)
            n = np.linalg.norm(v) + 1e-12
            vlad[i] = v / n

    vlad = vlad.reshape(-1)
    vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))          # power-law
    vlad /= (np.linalg.norm(vlad) + 1e-12)                # L2
    return vlad.astype(np.float32)


def bfs_hops(knn_ind: np.ndarray, goal_idx: int) -> np.ndarray:
    """
    在 kNN 无权图上从 goal 出发做 BFS，得到每个结点到 goal 的 hop 数。
    """
    n = knn_ind.shape[0]
    hops = np.full(n, np.iinfo(np.int32).max, dtype=np.int32)
    from collections import deque
    q = deque()
    hops[goal_idx] = 0
    q.append(goal_idx)
    while q:
        u = q.popleft()
        for v in knn_ind[u]:
            if hops[v] > hops[u] + 1:
                hops[v] = hops[u] + 1
                q.append(v)
    return hops


class AutoNavPlayer(Player):
    """
    只有自动导航的玩家：
    - EXPLORATION 阶段：不操作（依赖离线建图）
    - NAVIGATION 阶段：在线定位、朝目标推进、到达后自动 CHECKIN
    """
    def __init__(self, artifacts_dir: str = "./artifacts_baseline1234"):
        # 离线工件目录
        self.artifacts_dir = artifacts_dir

        # 固定使用同目录 target.jpg 作为目标
        self.fallback_target_path = (Path(__file__).resolve().parent / "target.jpg").as_posix()

        # 视觉与索引资源
        self.sift: Optional[cv2.SIFT] = None
        self.codebook = None
        self.idx2name: Optional[np.ndarray] = None
        self.vlads: Optional[np.ndarray] = None
        self.balltree: Optional[BallTree] = None
        self.knn_ind: Optional[np.ndarray] = None
        self.knn_dist: Optional[np.ndarray] = None
        self.meta = None
        self.img_root: Optional[Path] = None

        # 导航状态
        self.fpv: Optional[np.ndarray] = None
        self.goal_vec: Optional[np.ndarray] = None
        self.goal_idx: Optional[int] = None
        self.hops_to_goal: Optional[np.ndarray] = None
        self.last_cur_idx: Optional[int] = None
        self.last_goal_dist: float = math.inf
        self.checkin_sent: bool = False

        # 控制器超参数（可按赛道微调）
        self.forward_burst_steps: int = 4       # 发现变好后连续前进步数
        self.rotate_burst_steps: int = 6        # 变差卡住时原地转向步数
        self.stuck_patience: int = 5            # 连续变差多少步判定“卡住”
        self.dist_improve_eps: float = 1e-3     # 认为“变好”的最小改善量
        self.checkin_hop_thresh: int = 1        # hop≤阈值认为近旁
        self.checkin_dist_pct: float = 0.15     # 距离分位阈值（越小越严格）

        # 控制器内部计数器
        self._burst_left: int = 0
        self._rotate_left: int = 0
        self._rotate_dir_right: bool = True
        self._worsen_streak: int = 0

        super().__init__()

    # ---------- 生命周期 ----------
    def reset(self) -> None:
        # 每局重置临时状态
        self.fpv = None
        self.goal_vec = None
        self.goal_idx = None
        self.hops_to_goal = None
        self.last_cur_idx = None
        self.last_goal_dist = math.inf
        self.checkin_sent = False

        self._burst_left = 0
        self._rotate_left = 0
        self._rotate_dir_right = True
        self._worsen_streak = 0
        
        self._frame_count = 0
        self._last_action = Action.IDLE
        # 初始化 SIFT
        if self.sift is None:
            self.sift = cv2.SIFT_create()

    def pre_exploration(self) -> None:
        logging.info("pre exploration (auto navigator no-op)")

    def pre_navigation(self) -> None:
        """
        进入导航阶段时加载工件与目标。
        """
        super().pre_navigation()
        self._load_artifacts()
        self._prepare_goal_fixed_file()

    # ---------- 关键功能 ----------
    def _load_artifacts(self) -> None:
        """
        加载离线工件：meta.json, idx2name.npy, codebook.pkl, vlad.npy, knn_ind.npy, knn_dist.npy, balltree.pkl
        若无 balltree.pkl 则用 vlad.npy 重建。
        """
        art = Path(self.artifacts_dir)
        assert art.exists(), f"Artifacts dir not found: {art}"

        with open(art / "meta.json", "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.idx2name = np.load(art / "idx2name.npy", allow_pickle=True)
        with open(art / "codebook.pkl", "rb") as f:
            self.codebook = pickle.load(f)
        self.vlads = np.load(art / "vlad.npy")
        self.knn_ind = np.load(art / "knn_ind.npy")
        self.knn_dist = np.load(art / "knn_dist.npy")

        # BallTree
        pkl = art / "balltree.pkl"
        if pkl.exists():
            with open(pkl, "rb") as f:
                self.balltree = pickle.load(f)
        else:
            logging.warning("balltree.pkl not found; rebuilding BallTree from vlad.npy ...")
            self.balltree = BallTree(self.vlads, leaf_size=64)

        # 图像根目录
        self.img_root = Path(self.meta["image_root"]).resolve()
        logging.info(f"[META] N={self.meta['n_images']} VLAD_dim={self.meta['vlad_dim']} image_root={self.img_root}")

    def _prepare_goal_fixed_file(self) -> None:
        """
        不从 server 取目标；固定从同目录 target.jpg 读取，构造目标向量与 hop 距离。
        """
        tgt_path = Path(self.fallback_target_path)
        if not tgt_path.exists():
            raise FileNotFoundError(f"target.jpg not found at: {tgt_path}")

        img = cv2.imread(tgt_path.as_posix())
        if img is None:
            raise RuntimeError(f"target.jpg exists but cannot be read: {tgt_path}")

        # 使用单图目标的 VLAD；如需更鲁棒可加入旋转/模糊增强后取均值
        goal = vlad_encode(img, self.sift, self.codebook)
        goal /= (np.linalg.norm(goal) + 1e-12)
        self.goal_vec = goal.astype(np.float32)

        # 在数据库中找到目标最近邻结点
        dist, ind = self.balltree.query(self.goal_vec.reshape(1, -1), k=1)
        self.goal_idx = int(ind[0, 0])
        logging.info(f"[GOAL] goal_idx={self.goal_idx} (db match), dist={float(dist[0,0]):.4f}")

        # 预计算 hop（BFS）
        self.hops_to_goal = bfs_hops(self.knn_ind, self.goal_idx)

        # 统计“到目标距离”的分位线，确定 CHECKIN 阈值
        dists, _ = self.balltree.query(self.goal_vec.reshape(1, -1), k=min(64, len(self.vlads)))
        self._checkin_dist_abs = float(np.quantile(dists.reshape(-1), self.checkin_dist_pct))
        logging.info(f"[GOAL] check-in dist threshold ~= {self._checkin_dist_abs:.4f} (p{int(self.checkin_dist_pct*100)})")

        #dbg_goal = cv2.imread(str(self.img_root / self.idx2name[self.goal_idx]))
        #cv2.imshow("DB goal image", dbg_goal); cv2.waitKey(1)

    def _localize(self, fpv_bgr: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        把当前 FPV 编码为 VLAD，并在数据库中定位最近邻结点。
        返回：(cur_idx, dist_to_goal, vlad_vec)
        """
        v = vlad_encode(fpv_bgr, self.sift, self.codebook)
        # 距目标距离（L2）
        dist_goal = float(np.linalg.norm(v - self.goal_vec))
        # 当前数据库最近邻结点
        _, ind = self.balltree.query(v.reshape(1, -1), k=1)
        cur_idx = int(ind[0, 0])
        return cur_idx, dist_goal, v

    def _controller(self, cur_idx: int, dist_goal: float) -> Action:
        """
        简单反应式控制器：
        - hop 下降：连发 FORWARD
        - hop 不降且距离连续变差：短 burst 原地转向
        - 其余以 FORWARD 为主
        """
        # 到达判定：hop 足够小且距离也足够小 → CHECKIN
        cur_hop = int(self.hops_to_goal[cur_idx])
        if cur_hop <= self.checkin_hop_thresh and dist_goal <= self._checkin_dist_abs:
            if not self.checkin_sent:
                logging.info(f"[CHECKIN] hop={cur_hop}, dist={dist_goal:.4f} → CHECKIN")
                self.checkin_sent = True
                return Action.CHECKIN
            return Action.IDLE

        # 状态演化（改善/恶化）
        improved = (self.last_goal_dist - dist_goal) > self.dist_improve_eps
        if improved:
            self._worsen_streak = 0
            if self._burst_left == 0:  # 新一轮前进 burst
                self._burst_left = self.forward_burst_steps
        else:
            self._worsen_streak += 1

        # 若连续恶化，触发原地转向 burst（左右交替）
        if self._worsen_streak >= self.stuck_patience and self._rotate_left == 0:
            self._rotate_left = self.rotate_burst_steps
            self._rotate_dir_right = not self._rotate_dir_right

        # 优先执行旋转 burst
        if self._rotate_left > 0:
            self._rotate_left -= 1
            return Action.RIGHT if self._rotate_dir_right else Action.LEFT

        # 其次执行前进 burst
        if self._burst_left > 0:
            self._burst_left -= 1
            return Action.FORWARD

        # 否则以 FORWARD 为主；若定位未变则轻微转向
        if (self.last_cur_idx is not None) and (cur_idx == self.last_cur_idx):
            return Action.RIGHT if self._rotate_dir_right else Action.LEFT

        return Action.FORWARD

    # ---------- 回调 ----------
    def see(self, fpv: np.ndarray) -> None:
        if fpv is None or len(fpv.shape) != 3:
            return
        self.fpv = fpv

    def act(self) -> Action:
        # 尚未拿到状态或画面
        if self._state is None or self.fpv is None:
            return Action.IDLE

        phase = self._state[1]
        if phase == Phase.EXPLORATION:
            return Action.IDLE

        if phase == Phase.NAVIGATION:
            # 兜底：万一没准备好（通常 pre_navigation 会准备好）
            if self.balltree is None or self.goal_vec is None:
                self._load_artifacts()
                self._prepare_goal_fixed_file()

            # 在线定位 + 控制
            #cur_idx, dist_goal, _ = self._localize(self.fpv)
            #action = self._controller(cur_idx, dist_goal)

            #self.last_cur_idx = cur_idx
            #self.last_goal_dist = dist_goal
           #self._last_action = action

            cur_idx, dist_goal, v = self._localize(self.fpv)  # 注意接收 v
            hop_cur = int(self.hops_to_goal[cur_idx])         # 一定要用 cur_idx 取 hop
            action = self._controller(cur_idx, dist_goal)

# 记录
            self.last_cur_idx = cur_idx
            self.last_goal_dist = dist_goal
            self._last_action = action

        # 3) 【在这里插日志】——每 30 帧打一条
            if self._frame_count % 60 == 0:
             logging.info(
              f"[ACT] step={self._frame_count} cur={cur_idx} goal={self.goal_idx} "
              f"hop={hop_cur} dist={dist_goal:.4f} v_norm={np.linalg.norm(v):.4f}"
              )
        # 4) 计数 + 返回动作
            self._frame_count += 1
            return action

        return Action.IDLE


if __name__ == "__main__":
    import vis_nav_game
    vis_nav_game.play(the_player=AutoNavPlayer())
