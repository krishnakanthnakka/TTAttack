from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F

from attack_utils import (adv_attack_template, adv_attack_search, adv_attack_search_T,
                          add_gauss_noise, add_pulse_noise, adv_attack_template_S,
                          adv_attack_searchtemplate)
from data_utils import tensor2img
from pysot.core.config import cfg
from pysot.utils.anchor import Anchors
from pysot.tracker.base_tracker import SiameseTracker


class SiamRPNTracker(SiameseTracker):

    def __init__(self, model, dataset):
        super(SiamRPNTracker, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        hanning = np.hanning(self.score_size)

        # np.savetxt("hanning17.txt", hanning)
        # exit()

        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.anchors = self.generate_anchor(self.score_size)
        self.model = model
        self.model.eval()

    def generate_anchor(self, score_size):
        anchors = Anchors(cfg.ANCHOR.STRIDE,
                          cfg.ANCHOR.RATIOS,
                          cfg.ANCHOR.SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
            np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0] = xx.astype(np.float32)
        anchor[:, 1] = yy.astype(np.float32)
        return anchor

    def _convert_bbox(self, delta, anchor):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()
        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def get_z_crop(self, img, bbox):
        self.center_pos = np.array([bbox[0] + (bbox[2] - 1) / 2,
                                    bbox[1] + (bbox[3] - 1) / 2])
        self.size = np.array([bbox[2], bbox[3]])
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))
        self.channel_average = np.mean(img, axis=(0, 1))
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        return z_crop

    def get_x_crop(self, img):
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img,
                                    self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x),
                                    self.channel_average)
        return x_crop, scale_z

    def x_crop_2_res(self, img, x_crop, scale_z, target_bbox=None):
        outputs = self.model.track(x_crop)
        score = self._convert_score(outputs['cls'])  # (25x25x5,)
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)





        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0] * scale_z, self.size[1] * scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0] / self.size[1]) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))

        # r_c.fill(1)
        # s_c.fill(1)

        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        FORGET_PAST = False
        if FORGET_PAST:
            pscore = score

        best_idx = np.argmax(pscore)
        bbox = pred_bbox[:, best_idx] / scale_z
        bbox_search = pred_bbox[:, best_idx]
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR
        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        if target_bbox is not None:
            target_bbox /= scale_z
            target_bbox[0] += self.center_pos[0] - target_bbox[2] / 2
            target_bbox[1] += self.center_pos[1] - target_bbox[3] / 2
        if FORGET_PAST:
            lr = 1.0

        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr
        cx, cy, width, height = self._bbox_clip(cx, cy, width, height, img.shape[:2])
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])
        self.score = score
        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        best_score = score[best_idx]

        return {
            'bbox': bbox,
            'target_bbox_f': target_bbox,
            'best_score': best_score,
            'bbox_search': bbox_search,
            'best_score_post': pscore[best_idx],

            'zf': outputs['zf'][-1],
            'xf': outputs['xf'][-1]
            }

    def save_img(self, tensor_clean, tensor_adv, save_path, frame_id):
        img_clean = tensor2img(tensor_clean)
        cv2.imwrite(os.path.join(save_path, '%04d_clean.jpg' % frame_id), img_clean)
        img_adv = tensor2img(tensor_adv)
        cv2.imwrite(os.path.join(save_path, '%04d_adv.jpg' % frame_id), img_adv)
        tensor_diff = (tensor_adv - tensor_clean) * 10
        tensor_diff += 127.0
        img_diff = tensor2img(tensor_diff)
        cv2.imwrite(os.path.join(save_path, '%04d_diff.jpg' % frame_id), img_diff)

    def init(self, img, bbox):
        z_crop = self.get_z_crop(img, bbox)
        self.model.template(z_crop)
        self.template = z_crop
        self.lost = 0

    def init_adv(self, img, bbox, GAN, save_path=None, name=None):
        z_crop = self.get_z_crop(img, bbox)
        z_crop_adv = adv_attack_template(z_crop, GAN)
        self.model.template(z_crop_adv)
        if save_path is not None and name is not None:
            z_crop_img = tensor2img(z_crop)
            cv2.imwrite(os.path.join(save_path, name + '_clean.jpg'), z_crop_img)
            z_crop_adv_img = tensor2img(z_crop_adv)
            cv2.imwrite(os.path.join(save_path, name + '_adv.jpg'), z_crop_adv_img)
            diff = z_crop_adv - z_crop
            diff_img = tensor2img(diff)
            cv2.imwrite(os.path.join(save_path, name + '_diff.jpg'), diff_img)

    def init_adv_T(self, img, bbox, GAN, save_path=None, name=None, recompute=True):
        z_crop = self.get_z_crop(img, bbox)
        self.model.template(z_crop)
        if recompute:
            self.template = z_crop
            #print("Template updated")

        self.lost = 0
        self.longterm_state = False
        return tensor2img(z_crop)

    def init_adv_S(self, img, bbox, GAN, save_path=None, name=None):
        z_crop = self.get_z_crop(img, bbox)
        z_crop_adv = adv_attack_template_S(z_crop, GAN)
        self.model.template(z_crop_adv)
        if save_path is not None and name is not None:
            z_crop_img = tensor2img(z_crop)
            cv2.imwrite(os.path.join(save_path, name + '_clean.jpg'),
                        z_crop_img)
            z_crop_adv_img = tensor2img(z_crop_adv)
            cv2.imwrite(os.path.join(save_path, name + '_adv.jpg'),
                        z_crop_adv_img)
            diff = z_crop_adv - z_crop
            diff_img = tensor2img(diff)
            cv2.imwrite(os.path.join(save_path, name + '_diff.jpg'), diff_img)

    def track(self, img):
        x_crop, scale_z = self.get_x_crop(img)
        output_dict = self.x_crop_2_res(img, x_crop, scale_z)
        output_dict['cropx'] = x_crop

        return output_dict

    def track_adv(self, img, GAN, save_path=None, frame_id=None):
        x_crop, scale_z = self.get_x_crop(img)
        x_crop_adv = adv_attack_search(x_crop, GAN)
        output_dict = self.x_crop_2_res(img, x_crop_adv, scale_z)
        output_dict['cropx'] = x_crop_adv
        if save_path is not None and frame_id is not None:
            self.save_img(x_crop, x_crop_adv, save_path, frame_id)
        return output_dict

    def track_advjoint(self, img, GAN, save_path=None, frame_id=None):
        x_crop, scale_z = self.get_x_crop(img)
        x_crop_adv = adv_attack_searchtemplate(x_crop, self.template, GAN)
        output_dict = self.x_crop_2_res(img, x_crop_adv, scale_z)
        output_dict['cropx'] = x_crop_adv
        if save_path is not None and frame_id is not None:
            self.save_img(x_crop, x_crop_adv, save_path, frame_id)
        return output_dict

    def track_advT(self, img, GAN, dir_, save_path=None, frame_index=None):
        x_crop, scale_z = self.get_x_crop(img)
        x_crop_adv, perturbmetrics = adv_attack_search_T(x_crop, self.template, GAN, dir_, frame_index)
        output_dict = self.x_crop_2_res(img, x_crop_adv, scale_z, perturbmetrics['target_bbox'])

        output_dict.update(perturbmetrics)

        # bbox_search = output_dict['bbox_search']
        # offset = cfg.TRACK.INSTANCE_SIZE / 2
        # x_crop_j = x_crop_adv[0].detach().cpu().numpy()
        # x_crop_j = np.ascontiguousarray(x_crop_j.transpose(1, 2, 0))
        # x_crop_j = x_crop_j.astype('uint8')

        '''
        cv2.rectangle(x_crop_j, (int(offset + bbox_search[0] - bbox_search[2] / 2),
                                 int(offset + bbox_search[1] - bbox_search[3] / 2)),
                      (int(offset + bbox_search[0] + bbox_search[2] / 2),
                       int(offset + bbox_search[1] + bbox_search[3] / 2)),
                      (128, 128, 0),
                      3)

        cv2.putText(x_crop_j, str("{:.2f}, {:.2f}".format(output_dict['best_score'],
                                                          output_dict['best_score_post'])),
                    (int(offset + bbox_search[0] - bbox_search[2] / 2),
                     int(offset + bbox_search[1] - bbox_search[3] / 2) - 10),
                    0, 0.4, (0, 0, 255))
        '''

        # x_crop_adv[0] = torch.tensor(x_crop_j.transpose(2, 0, 1))
        # output_dict['cropx'] = torch.cat((x_crop_adv, x_crop), 0)
        # perturb_img = torch.abs(x_crop_adv - x_crop) + 127
        # output_dict['perturb'] = torch.cat((perturb_img, x_crop), 0)

        # not needed below
        #output_dict['cropx'] = torch.cat((perturb_img, x_crop), 0)

        output_dict['lost'] = 0
        if 0 and save_path is not None and frame_id is not None:
            self.save_img(x_crop, x_crop_adv, save_path, frame_id)
        return output_dict

    def track_gauss(self, img, sigma, save_path=None, frame_id=None):
        x_crop, scale_z = self.get_x_crop(img)
        x_crop_adv = add_gauss_noise(x_crop, sigma)
        output_dict = self.x_crop_2_res(img, x_crop_adv, scale_z)
        if save_path is not None and frame_id is not None:
            self.save_img(x_crop, x_crop_adv, save_path, frame_id)
        return output_dict

    def track_impulse(self, img, prob, save_path, frame_id):
        x_crop, scale_z = self.get_x_crop(img)
        x_crop_adv = add_pulse_noise(x_crop, prob)
        output_dict = self.x_crop_2_res(img, x_crop_adv, scale_z)
        if save_path is not None and frame_id is not None:
            self.save_img(x_crop, x_crop_adv, save_path, frame_id)
        return output_dict

    def track_heatmap(self, img):
        x_crop, scale_z = self.get_x_crop(img)
        output_dict = self.x_crop_2_res(img, x_crop, scale_z)
        score_map = np.max(self.score.reshape(5, 25, 25), axis=0)
        return output_dict, score_map

    def track_supp(self, img, GAN, save_path, frame_id):
        x_crop, scale_z = self.get_x_crop(img)
        x_crop_img = tensor2img(x_crop)
        cv2.imwrite(os.path.join(save_path, 'ori_search_%d.jpg' % frame_id), x_crop_img)
        outputs_clean = self.model.track(x_crop)
        score = self._convert_score(outputs_clean['cls'])  # (25x25x5,)
        heatmap_clean = 255.0 * np.max(score.reshape(5, 25, 25), axis=0)  # [0,1]
        heatmap_clean = cv2.resize(heatmap_clean, (255, 255), interpolation=cv2.INTER_CUBIC)
        heatmap_clean = cv2.applyColorMap(heatmap_clean.clip(
            0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(save_path, 'heatmap_clean_%d.jpg' % frame_id), heatmap_clean)
        x_crop_adv = adv_attack_search(x_crop, GAN)
        output_dict = self.x_crop_2_res(img, x_crop_adv, scale_z)
        x_crop_img_adv = tensor2img(x_crop_adv)
        cv2.imwrite(os.path.join(save_path, 'adv_search_%d.jpg' % frame_id), x_crop_img_adv)
        score_adv = self.score
        heatmap_adv = 255.0 * np.max(score_adv.reshape(5, 25, 25), axis=0)
        heatmap_adv = cv2.resize(heatmap_adv, (255, 255), interpolation=cv2.INTER_CUBIC)
        heatmap_adv = cv2.applyColorMap(heatmap_adv.clip(0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(save_path, 'heatmap_adv_%d.jpg' % frame_id), heatmap_adv)
        return output_dict

    # def track_adv_targeted(self, img):

    #     # exit()
    #     x_crop, scale_z = self.get_x_crop(img)

    #     output_dict = self.x_crop_2_res_adv(img, x_crop, scale_z)
    #     cx, cy, w, h = output_dict['bbox_search']
    #     x1, y1, x2, y2 = center2corner([cx, cy, w, h])
    #     target = x1, y1, x2, y2
    #     print("Pred: ", target)
    #     size = self.score_size
    #     anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
    #     cls = 0 * np.ones((anchor_num, size, size), dtype=np.int64)
    #     delta = np.zeros((4, anchor_num, size, size), dtype=np.float32)
    #     delta_weight = np.zeros((anchor_num, size, size), dtype=np.float32)

    #     def select(position, keep_num=16):
    #         num = position[0].shape[0]
    #         if num <= keep_num:
    #             return position, num
    #         slt = np.arange(num)
    #         np.random.shuffle(slt)
    #         slt = slt[:keep_num]
    #         return tuple(p[slt] for p in position), keep_num

    #     tcx, tcy, tw, th = corner2center(target)
    #     anchor_box = self.anchors2.all_anchors[0]
    #     anchor_center = self.anchors2.all_anchors[1]

    #     # anchor_s1 = anchor_box.reshape(4, -1).T
    #     # anchor_s = anchor_center.reshape(4, -1).T
    #     # np.savetxt("anchors2.txt", anchor_s, fmt="%d")
    #     # np.savetxt("anchors23.txt", anchor_s1, fmt="%d")

    #     x1, y1, x2, y2 = anchor_box[0], anchor_box[1], anchor_box[2], anchor_box[3]
    #     cx, cy, w, h = anchor_center[0], anchor_center[1], anchor_center[2], anchor_center[3]

    #     print(np.max(x2))

    #     delta[0] = (tcx - cx) / w
    #     delta[1] = (tcy - cy) / h
    #     delta[2] = np.log(tw / w)
    #     delta[3] = np.log(th / h)

    #     overlap = IoU([x1, y1, x2, y2], target)

    #     cfg.TRAIN.LOW = 0.05

    #     pos = np.where(overlap > cfg.TRAIN.THR_HIGH)
    #     neg = np.where(overlap < cfg.TRAIN.THR_LOW)

    #     neg, neg_num = select(neg, 1)

    #     print("Num of proposals:", neg_num)
    #     print("overlap with target:{}".format(overlap[neg]))

    #     cls[neg] = 1
    #     delta_weight[neg] = 1. / (neg_num + 1e-6)

    #     bbox_target = [int(anchor_box[i, neg[0], neg[1], neg[2]][0]) for i in range(4)]
    #     # bbox_target[0] += 127.5
    #     # bbox_target[1] += 127.5
    #     output_dict['bbox_target'] = bbox_target

    #     print("Adv Traget:", output_dict['bbox_target'])
    #     x_crop_img = output_dict['x_crop_img']

    #     offset = cfg.TRACK.INSTANCE_SIZE / 2

    #     cv2.rectangle(x_crop_img, (int(offset + bbox_target[0]), int(offset + bbox_target[1])),
    #                   (int(offset + bbox_target[2]), int(offset + bbox_target[3])), (0, 255, 0), 3)

    #     cv2.imwrite("search_target_clean.png", x_crop_img)

    #     #deltaperturb = self.perturbation
    #     deltaperturb = torch.zeros_like(x_crop)

    #     print("Initial perturbnorm:{}".format(torch.max(abs(deltaperturb))))

    #     deltaperturb.requires_grad_()
    #     TARGETED = True

    #     eps_iter = 2.0
    #     eps = 8.0
    #     clip_min, clip_max = 0.0, 255.0

    #     nb_iter = 50

    #     label_cls = torch.from_numpy(cls).cuda()
    #     label_loc = torch.from_numpy(delta).cuda()
    #     label_loc_weight = torch.from_numpy(delta_weight).cuda()

    #     for ii in range(nb_iter):
    #         outputs = self.model.forward_adv(x_crop + deltaperturb, label_cls, label_loc, label_loc_weight)
    #         loss = outputs['total_loss']
    #         print(ii, loss.data, outputs['cls_loss'].data, outputs['loc_loss'].data)

    #         if TARGETED:
    #             loss = -loss
    #         loss.backward(retain_graph=True)
    #         grad_sign = deltaperturb.grad.data.sign()
    #         deltaperturb.data = deltaperturb.data + batch_multiply(eps_iter, grad_sign)
    #         deltaperturb.data = batch_clamp(eps, deltaperturb.data)
    #         deltaperturb.data = clamp(x_crop.data + deltaperturb.data, clip_min, clip_max
    #                                   ) - x_crop.data

    #         deltaperturb.grad.data.zero_()

    #     self.perturbation = deltaperturb

    #     x_crop_adv = clamp(x_crop + deltaperturb, clip_min, clip_max)
    #     output_dict = self.x_crop_2_res(img, x_crop_adv, scale_z)
    #     output_dict['cropx'] = x_crop
    #     output_dict['search_target'] = x_crop_img

    #     #save_image(x_crop_adv / 255, "search_target_adv.png")
    #     # exit()

    #     return output_dict
