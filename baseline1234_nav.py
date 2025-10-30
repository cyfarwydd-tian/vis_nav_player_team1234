# player_nav_vlad.py
from vis_nav_game import Player, Action, Phase
import pygame
import cv2
import os
import pickle
import json
from pathlib import Path

import numpy as np
from natsort import natsorted
from sklearn.neighbors import BallTree


class KeyboardPlayerVLAD(Player):
    def __init__(self):
        super().__init__()
        self.fpv = None
        self.screen = None
        self.last_act = Action.IDLE
        self.keymap = None

        # paths
        self.img_dir = "./data/images_subsample"
        self.art_dir = "./data/artfact1"

        # feature stuff
        self.sift = cv2.SIFT_create()
        self.codebook = None
        self.vlad_db = None
        self.tree = None
        self.idx2name = None

        self.goal_id = None  # index in db

        self._load_artifacts()

    # ===================== artifact loading =====================
    def _load_artifacts(self):
        art = Path(self.art_dir)
        if not art.exists():
            raise FileNotFoundError(f"{self.art_dir} not found. Run train_exploration_vlad.py first.")

        # load codebook
        with open(art / "codebook.pkl", "rb") as f:
            self.codebook = pickle.load(f)

        # load vlad db
        self.vlad_db = np.load(art / "vlad.npy")

        # load balltree
        with open(art / "balltree.pkl", "rb") as f:
            self.tree = pickle.load(f)

        # load idx2name
        self.idx2name = np.load(art / "idx2name.npy")

        # meta just for debug
        meta_path = art / "meta.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
            print("Loaded artifacts:", meta)
        else:
            print("Artifacts loaded.")

    # ===================== basic player stuff =====================
    def reset(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None

        pygame.init()
        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT,
        }

    def act(self):
        """
        Handle player actions based on keyboard input
        """
        for event in pygame.event.get():
            #  Quit if user closes window or presses escape
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT
            # Check if a key has been pressed
            if event.type == pygame.KEYDOWN:
                # Check if the pressed key is in the keymap
                if event.key in self.keymap:
                    # If yes, bitwise OR the current action with the new one
                    # This allows for multiple actions to be combined into a single action
                    self.last_act |= self.keymap[event.key]
                else:
                    # If a key is pressed that is not mapped to an action, then display target images
                    self.show_target_images()
            # Check if a key has been released
            if event.type == pygame.KEYUP:
                # Check if the released key is in the keymap
                if event.key in self.keymap:
                    # If yes, bitwise XOR the current action with the new one
                    # This allows for updating the accumulated actions to reflect the current sate of the keyboard inputs accurately
                    self.last_act ^= self.keymap[event.key]
        return self.last_act
    def show_target_images(self):
        """
        Display front, right, back, and left views of target location in 2x2 grid manner
        """
        targets = self.get_target_images()

        # Return if the target is not set yet
        if targets is None or len(targets) <= 0:
            return

        # Create a 2x2 grid of the 4 views of target location
        hor1 = cv2.hconcat(targets[:2])
        hor2 = cv2.hconcat(targets[2:])
        concat_img = cv2.vconcat([hor1, hor2])

        w, h = concat_img.shape[:2]
        
        color = (0, 0, 0)

        concat_img = cv2.line(concat_img, (int(h/2), 0), (int(h/2), w), color, 2)
        concat_img = cv2.line(concat_img, (0, int(w/2)), (h, int(w/2)), color, 2)

        w_offset = 25
        h_offset = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        line = cv2.LINE_AA
        size = 0.75
        stroke = 1

        cv2.putText(concat_img, 'Front View', (h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Right View', (int(h/2) + h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Back View', (h_offset, int(w/2) + w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Left View', (int(h/2) + h_offset, int(w/2) + w_offset), font, size, color, stroke, line)

        cv2.imshow(f'KeyboardPlayer:target_images', concat_img)
        cv2.waitKey(1)

    def set_target_images(self, images):
        """
        Set target images
        """
        super(KeyboardPlayerVLAD, self).set_target_images(images)
        self.show_target_images()
    # ===================== feature utils =====================
    def _img_vlad(self, img_bgr):
        _, des = self.sift.detectAndCompute(img_bgr, None)
        if des is None or len(des) == 0:
            k = self.codebook.n_clusters
            d = self.codebook.cluster_centers_.shape[1]
            v = np.zeros(k * d, dtype=np.float32)
            return v
        pred = self.codebook.predict(des)
        centers = self.codebook.cluster_centers_
        k = self.codebook.n_clusters
        vlad = np.zeros((k, des.shape[1]), dtype=np.float32)
        for i in range(k):
            mask = (pred == i)
            if np.any(mask):
                vlad[i] = np.sum(des[mask] - centers[i], axis=0)
        vlad = vlad.reshape(-1)
        vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))
        n = np.linalg.norm(vlad)
        if n > 1e-12:
            vlad = vlad / n
        return vlad

    def _nearest_id(self, img_bgr, k=1):
        v = self._img_vlad(img_bgr).reshape(1, -1)
        dist, idx = self.tree.query(v, k)
        return idx[0][0], dist[0][0]

    # ===================== hint logic =====================
    def _show_hint_window(self, cur_id, goal_id, step=3):
        """show 2x2 grid: current match, suggested, target front, and text overlay"""
        # decide suggested id
        # if goal is ahead -> move forward a bit
        if goal_id is not None:
            if goal_id > cur_id:
                sug_id = min(goal_id, cur_id + step)
            else:
                sug_id = max(goal_id, cur_id - step)
        else:
            sug_id = cur_id + step

        def _read_by_dbid(dbid):
            if dbid < 0 or dbid >= len(self.idx2name):
                return np.zeros((240, 320, 3), dtype=np.uint8)
            fname = self.idx2name[dbid]
            path = os.path.join(self.img_dir, fname)
            if os.path.exists(path):
                return cv2.imread(path)
            return np.zeros((240, 320, 3), dtype=np.uint8)

        cur_img = _read_by_dbid(cur_id)
        sug_img = _read_by_dbid(sug_id)

        # goal: use target front image if we have it
        targets = self.get_target_images()
        if targets is not None and len(targets) > 0:
            goal_img = targets[0]
        else:
            goal_img = _read_by_dbid(goal_id) if goal_id is not None else np.zeros_like(cur_img)

        # make same size
        def _resize(x, w=320, h=240):
            return cv2.resize(x, (w, h)) if x is not None else np.zeros((h, w, 3), dtype=np.uint8)

        cur_img = _resize(cur_img)
        sug_img = _resize(sug_img)
        goal_img = _resize(goal_img)
        blank = np.zeros_like(cur_img)

        top = cv2.hconcat([cur_img, sug_img])
        bottom = cv2.hconcat([goal_img, blank])
        grid = cv2.vconcat([top, bottom])

        # draw lines
        H, W = grid.shape[:2]
        grid = cv2.line(grid, (W // 2, 0), (W // 2, H), (0, 0, 0), 2)
        grid = cv2.line(grid, (0, H // 2), (W, H // 2), (0, 0, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(grid, "Current match", (10, 25), font, 0.7, (0, 0, 0), 2)
        cv2.putText(grid, "Suggested view", (W // 2 + 10, 25), font, 0.7, (0, 0, 0), 2)
        cv2.putText(grid, "Target(front)", (10, H // 2 + 25), font, 0.7, (0, 0, 0), 2)
        if goal_id is not None:
            cv2.putText(grid, f"cur={cur_id}  goal={goal_id}", (10, H - 15), font, 0.6, (50, 50, 50), 2)

        cv2.imshow("Navigation hint", grid)
        cv2.waitKey(1)

    def _english_hint_text(self, cur_id, goal_id, dist_val):
        if goal_id is None:
            return "Goal is not set yet. Look around to see the target first."
        gap = goal_id - cur_id
        abs_gap = abs(gap)
        if abs_gap <= 2:
            return "You are very close to the goal. Press SPACE to check in."
        if gap > 0:
            return f"Goal is ahead in the sequence. Move forward. (gap={abs_gap})"
        else:
            return f"You have passed the goal. Turn back / move backward. (gap={abs_gap})"

    # ===================== main vision hook =====================
    def see(self, fpv):
        if fpv is None or len(fpv.shape) < 3:
            return
        self.fpv = fpv

        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))
            pygame.display.set_caption("KeyboardPlayerVLAD:fpv")

        # game state is available?
        if self._state:
            phase = self._state[1]
            # navigation phase
            if phase == Phase.NAVIGATION:
                # ensure goal
                if self.goal_id is None:
                    targets = self.get_target_images()
                    if targets is not None and len(targets) > 0:
                        # just use the front view to set goal
                        tgt_front = targets[0]
                        gid, _ = self._nearest_id(tgt_front)
                        self.goal_id = gid
                        print(f"[NAV] Goal ID set to {self.goal_id}")

                # if user pressed Q => give hint
                keys = pygame.key.get_pressed()
                if keys[pygame.K_q]:
                    cur_id, dist_val = self._nearest_id(self.fpv)
                    msg = self._english_hint_text(cur_id, self.goal_id, dist_val)
                    print("[HINT]", msg)
                    self._show_hint_window(cur_id, self.goal_id, step=3)

        # draw fpv
        rgb = fpv[:, :, ::-1]  # bgr->rgb
        surf = pygame.image.frombuffer(rgb.tobytes(), (rgb.shape[1], rgb.shape[0]), "RGB")
        self.screen.blit(surf, (0, 0))
        pygame.display.update()


if __name__ == "__main__":
    import vis_nav_game
    vis_nav_game.play(the_player=KeyboardPlayerVLAD())
