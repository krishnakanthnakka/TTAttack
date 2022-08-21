from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch

from pysot.core.config import cfg
from pysot.tracker.siamrpn_tracker import SiamRPNTracker
from attack_utils import adv_attack_template, adv_attack_search, adv_attack_search_T


class SiamRPNLTTracker(SiamRPNTracker):

    def __init__(self, model, dataset=None):
        super(SiamRPNLTTracker, self).__init__(model, dataset)
        self.longterm_state = False

    def track(self, img):
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z

        if self.longterm_state:
            instance_size = cfg.TRACK.LOST_INSTANCE_SIZE
        else:
            instance_size = cfg.TRACK.INSTANCE_SIZE

        score_size = (instance_size - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        hanning = np.hanning(score_size)
        window = np.outer(hanning, hanning)
        window = np.tile(window.flatten(), self.anchor_num)
        anchors = self.generate_anchor(score_size)
        s_x = s_z * (instance_size / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos, instance_size, round(s_x),
                                    self.channel_average)
        outputs = self.model.track(x_crop)
        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :])
                     / (sz(self.size[0] * scale_z, self.size[1] * scale_z)))
        # ratio penalty
        r_c = change((self.size[0] / self.size[1])
                     / (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window
        if not self.longterm_state:
            pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                window * cfg.TRACK.WINDOW_INFLUENCE
        else:
            pscore = pscore * (1 - 0.001) + window * 0.001
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        best_score = score[best_idx]
        if best_score >= cfg.TRACK.CONFIDENCE_LOW:
            cx = bbox[0] + self.center_pos[0]
            cy = bbox[1] + self.center_pos[1]

            width = self.size[0] * (1 - lr) + bbox[2] * lr
            height = self.size[1] * (1 - lr) + bbox[3] * lr
        else:
            cx = self.center_pos[0]
            cy = self.center_pos[1]

            width = self.size[0]
            height = self.size[1]

        cx, cy, width, height = self._bbox_clip(cx, cy, width, height, img.shape[:2])

        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])
        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        if best_score < cfg.TRACK.CONFIDENCE_LOW:
            self.longterm_state = True
        elif best_score > cfg.TRACK.CONFIDENCE_HIGH:
            self.longterm_state = False

        output_dict = {
            'bbox': bbox,
            'best_score': best_score,
            'lost': self.lost,
            'cropx': x_crop
        }

        perturbmetrics = {"metrics": {"MAE": torch.tensor(0.0), "SSIM": 100}}
        output_dict.update(perturbmetrics)

        return output_dict

    def track_adv(self, img, GAN):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        if self.longterm_state:
            instance_size = cfg.TRACK.LOST_INSTANCE_SIZE
            self.lost += 1
        else:
            instance_size = cfg.TRACK.INSTANCE_SIZE

        score_size = (instance_size - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        hanning = np.hanning(score_size)
        window = np.outer(hanning, hanning)
        window = np.tile(window.flatten(), self.anchor_num)
        anchors = self.generate_anchor(score_size)
        s_x = s_z * (instance_size / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos, instance_size,
                                    round(s_x), self.channel_average)
        x_crop = adv_attack_search(x_crop, GAN, search_sz=(instance_size, instance_size))
        outputs = self.model.track(x_crop)
        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :])
                     / (sz(self.size[0] * scale_z, self.size[1] * scale_z)))
        # ratio penalty
        r_c = change((self.size[0] / self.size[1])
                     / (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window
        if not self.longterm_state:
            pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                window * cfg.TRACK.WINDOW_INFLUENCE
        else:
            pscore = pscore * (1 - 0.001) + window * 0.001
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        best_score = score[best_idx]
        if best_score >= cfg.TRACK.CONFIDENCE_LOW:
            cx = bbox[0] + self.center_pos[0]
            cy = bbox[1] + self.center_pos[1]

            width = self.size[0] * (1 - lr) + bbox[2] * lr
            height = self.size[1] * (1 - lr) + bbox[3] * lr
        else:
            cx = self.center_pos[0]
            cy = self.center_pos[1]

            width = self.size[0]
            height = self.size[1]

        cx, cy, width, height = self._bbox_clip(cx, cy, width, height, img.shape[:2])

        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])
        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        if best_score < cfg.TRACK.CONFIDENCE_LOW:
            self.longterm_state = True
        elif best_score > cfg.TRACK.CONFIDENCE_HIGH:
            self.longterm_state = False

        output_dict = {
            'bbox': bbox,
            'best_score': best_score,
            'lost': self.lost,
            'cropx': x_crop
        }

        output_dict.update(perturbmetrics)
        return output_dict

    def track_advT(self, img, GAN, dir_, enhance=False):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z

        if self.longterm_state:
            instance_size = cfg.TRACK.LOST_INSTANCE_SIZE
            self.lost += 1
        else:
            instance_size = cfg.TRACK.INSTANCE_SIZE

        score_size = (instance_size - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE

        hanning = np.hanning(score_size)
        window = np.outer(hanning, hanning)
        window = np.tile(window.flatten(), self.anchor_num)
        anchors = self.generate_anchor(score_size)
        s_x = s_z * (instance_size / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos, instance_size,
                                    round(s_x), self.channel_average)
        x_crop, perturbmetrics = adv_attack_search_T(x_crop, self.template, GAN, dir_,
                                                     (instance_size, instance_size), enhance)

        outputs = self.model.track(x_crop)
        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], anchors)
        target_bbox = perturbmetrics['target_bbox']

        if target_bbox is not None:
            target_bbox /= scale_z
            target_bbox[0] += self.center_pos[0] - target_bbox[2] / 2
            target_bbox[1] += self.center_pos[1] - target_bbox[3] / 2

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :])
                     / (sz(self.size[0] * scale_z, self.size[1] * scale_z)))
        # ratio penalty
        r_c = change((self.size[0] / self.size[1])
                     / (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window
        if not self.longterm_state:
            pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + window * cfg.TRACK.WINDOW_INFLUENCE
        else:
            pscore = pscore * (1 - 0.001) + window * 0.001
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        best_score = score[best_idx]
        if best_score >= cfg.TRACK.CONFIDENCE_LOW:
            cx = bbox[0] + self.center_pos[0]
            cy = bbox[1] + self.center_pos[1]

            width = self.size[0] * (1 - lr) + bbox[2] * lr
            height = self.size[1] * (1 - lr) + bbox[3] * lr
        else:
            cx = self.center_pos[0]
            cy = self.center_pos[1]

            width = self.size[0]
            height = self.size[1]

        cx, cy, width, height = self._bbox_clip(cx, cy, width, height, img.shape[:2])
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        if best_score < cfg.TRACK.CONFIDENCE_LOW:
            self.longterm_state = True
        elif best_score > cfg.TRACK.CONFIDENCE_HIGH:
            self.longterm_state = False

        output_dict = {
            'bbox': bbox,
            'best_score': best_score,
            'target_bbox_f': target_bbox,
            'lost': self.lost,
            'cropx': x_crop
        }

        output_dict.update(perturbmetrics)
        return output_dict
