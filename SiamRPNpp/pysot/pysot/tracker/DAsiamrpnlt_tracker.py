# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import torch.nn.functional as F
import torch
from pysot.core.config import cfg
from pysot.utils.anchor import Anchors
from pysot.tracker.base_tracker import SiameseTracker
from data_utils import tensor2img
import cv2
import os
''' Adversarial Attack tools :)'''
from attack_utils import adv_attack_template, adv_attack_search, adv_attack_search_T, \
    add_gauss_noise, add_pulse_noise, adv_attack_template_S

import torch.nn as nn
import torch.nn.functional as F


class SiamRPN(nn.Module):
    def __init__(self, size=2, feature_out=512, anchor=5):
        configs = [3, 96, 256, 384, 384, 256]
        configs = list(map(lambda x: 3 if x == 3 else x * size, configs))
        feat_in = configs[-1]
        super(SiamRPN, self).__init__()
        self.featureExtract = nn.Sequential(
            nn.Conv2d(configs[0], configs[1], kernel_size=11, stride=2),
            nn.BatchNorm2d(configs[1]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[1], configs[2], kernel_size=5),
            nn.BatchNorm2d(configs[2]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[2], configs[3], kernel_size=3),
            nn.BatchNorm2d(configs[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[3], configs[4], kernel_size=3),
            nn.BatchNorm2d(configs[4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[4], configs[5], kernel_size=3),
            nn.BatchNorm2d(configs[5]),
        )

        self.anchor = anchor
        self.feature_out = feature_out

        self.conv_r1 = nn.Conv2d(feat_in, feature_out * 4 * anchor, 3)
        self.conv_r2 = nn.Conv2d(feat_in, feature_out, 3)
        self.conv_cls1 = nn.Conv2d(feat_in, feature_out * 2 * anchor, 3)
        self.conv_cls2 = nn.Conv2d(feat_in, feature_out, 3)
        self.regress_adjust = nn.Conv2d(4 * anchor, 4 * anchor, 1)

        self.r1_kernel = []
        self.cls1_kernel = []

        self.cfg = {}

    def forward(self, x):
        x_f = self.featureExtract(x)
        return self.regress_adjust(F.conv2d(self.conv_r2(x_f), self.r1_kernel)), \
            F.conv2d(self.conv_cls2(x_f), self.cls1_kernel)

    def temple(self, z):
        z_f = self.featureExtract(z)
        r1_kernel_raw = self.conv_r1(z_f)
        cls1_kernel_raw = self.conv_cls1(z_f)
        kernel_size = r1_kernel_raw.data.size()[-1]
        self.r1_kernel = r1_kernel_raw.view(self.anchor * 4, self.feature_out, kernel_size, kernel_size)
        self.cls1_kernel = cls1_kernel_raw.view(self.anchor * 2, self.feature_out, kernel_size, kernel_size)


class SiamRPNBIG(SiamRPN):
    def __init__(self):
        super(SiamRPNBIG, self).__init__(size=2)
        self.cfg = {'lr': 0.295, 'window_influence': 0.42, 'penalty_k': 0.055,
                    'instance_size': 255, 'adaptive': True}  # 0.383


class SiamRPNvotLT(SiamRPN):
    def __init__(self):
        super(SiamRPNvotLT, self).__init__(size=1, feature_out=256)
        self.cfg = {'lr': 0.45, 'window_influence': 0.44, 'penalty_k': 0.04,
                    'instance_size': 255, 'adaptive': True}  # 0.355


class SiamRPNvot(SiamRPN):
    def __init__(self):
        super(SiamRPNvot, self).__init__(size=1, feature_out=256)
        self.cfg = {'lr': 0.45, 'window_influence': 0.44, 'penalty_k': 0.04,
                    'instance_size': 255, 'adaptive': False}  # 0.355


class SiamRPNotb(SiamRPN):
    def __init__(self):
        super(SiamRPNotb, self).__init__(size=1, feature_out=256)
        self.cfg = {'lr': 0.30, 'window_influence': 0.40, 'penalty_k': 0.22,
                    'instance_size': 255, 'adaptive': False}  # 0.655


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    return img


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans, out_mode='torch', new=False):

    if isinstance(pos, float):
        pos = [pos, pos]
    sz = original_sz
    im_sz = im.shape
    c = (original_sz + 1) / 2
    context_xmin = round(pos[0] - c)  # floor(pos(2) - sz(2) / 2);
    context_xmax = context_xmin + sz - 1
    context_ymin = round(pos[1] - c)  # floor(pos(1) - sz(1) / 2);
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    # zzp: a more easy speed version
    r, c, k = im.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k),
                         np.uint8)  # 0 is better than 1 initialization
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
    else:
        im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))  # zzp: use cv to get a better speed
    else:
        im_patch = im_patch_original

    return im_to_torch(im_patch) if out_mode in 'torch' else im_patch


class DASiamRPNLTTracker(SiameseTracker):
    def __init__(self, model, dataset):
        super(DASiamRPNLTTracker, self).__init__()
        self.windowing = 'cosine'  # to penalize large displacements [cosine/uniform]

        if dataset in ['OTB2', 'OTB100', 'UAV123']:
            net = SiamRPNotb()
            net.load_state_dict(torch.load(
                '/cvlabdata1/home/krishna/AttTracker/GIT/CSA/pysot/experiments/DAsiamrpn_alex_dwxcorr/modelOTB.pth'))

        elif dataset in ['VOT2018', 'VOT2018-LT']:
            net = SiamRPNvot()

            net.load_state_dict(torch.load(
                '/cvlabdata1/home/krishna/AttTracker/GIT/CSA/pysot/experiments/DAsiamrpn_alex_dwxcorr/modelVOT.pth'))

        else:
            assert False, 'No tracker found for given dataset'

        # net.load_state_dict(torch.load(
        #     '/cvlabdata1/home/krishna/AttTracker/GIT/CSA/pysot/experiments/DAsiamrpn_alex_dwxcorr/model.pth'))

        self.cfg = cfg

        self.net = net.cuda().eval()

        self.INSTANCE_SIZE = net.cfg['instance_size']
        self.total_stride = cfg.ANCHOR.STRIDE

        self.cfg.TRACK.LR = net.cfg['lr']
        self.cfg.TRACK.WINDOW_INFLUENCE = net.cfg['window_influence']
        self.cfg.TRACK.PENALTY_K = net.cfg['penalty_k']
        self.cfg.TRACK.ADAPTIVE = net.cfg['adaptive']
        self.longterm_state = False
        self.score_size = (self.INSTANCE_SIZE - self.cfg.TRACK.EXEMPLAR_SIZE) / self.total_stride + 1

    def generate_anchor(self, total_stride, scales, ratios, score_size):
        anchor_num = len(ratios) * len(scales)
        anchor = np.zeros((anchor_num, 4), dtype=np.float32)
        size = total_stride * total_stride
        count = 0
        for ratio in ratios:
            # ws = int(np.sqrt(size * 1.0 / ratio))
            ws = int(np.sqrt(size / ratio))
            hs = int(ws * ratio)
            for scale in scales:
                wws = ws * scale
                hhs = hs * scale
                anchor[count, 0] = 0
                anchor[count, 1] = 0
                anchor[count, 2] = wws
                anchor[count, 3] = hhs
                count += 1

        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size / 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
            np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def tracker_eval(self, net, x_crop, target_pos, target_sz, window, scale_z):
        delta, score = net(x_crop)

        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()
        score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1), dim=0).data[1, :].cpu().numpy()

        if self.longterm_state:
            instance_size = cfg.TRACK.LOST_INSTANCE_SIZE  # small object big search region
        else:
            instance_size = cfg.TRACK.INSTANCE_SIZE

        self.score_size = (instance_size - self.cfg.TRACK.EXEMPLAR_SIZE) / self.total_stride + 1

        self.anchor = self.generate_anchor(self.total_stride, self.cfg.ANCHOR.SCALES,
                                           self.cfg.ANCHOR.RATIOS, int(self.score_size))

        if self.windowing == 'cosine':
            window = np.outer(np.hanning(self.score_size), np.hanning(self.score_size))
        elif self.windowing == 'uniform':
            window = np.ones((self.score_size, self.score_size))
        window = np.tile(window.flatten(), self.cfg.ANCHOR.ANCHOR_NUM)

        delta[0, :] = delta[0, :] * self.anchor[:, 2] + self.anchor[:, 0]
        delta[1, :] = delta[1, :] * self.anchor[:, 3] + self.anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * self.anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * self.anchor[:, 3]

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            sz2 = (w + pad) * (h + pad)
            return np.sqrt(sz2)

        def sz_wh(wh):
            pad = (wh[0] + wh[1]) * 0.5
            sz2 = (wh[0] + pad) * (wh[1] + pad)
            return np.sqrt(sz2)

        # size penalty
        s_c = change(sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz)))  # scale penalty
        r_c = change((target_sz[0] / target_sz[1]) / (delta[2, :] / delta[3, :]))  # ratio penalty

        penalty = np.exp(-(r_c * s_c - 1.) * self.cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window
        if not self.longterm_state:
            pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                window * cfg.TRACK.WINDOW_INFLUENCE
        else:
            pscore = pscore * (1 - 0.001) + window * 0.001

        # window float
#        pscore = pscore * (1 - self.cfg.TRACK.WINDOW_INFLUENCE) + window * self.cfg.TRACK.WINDOW_INFLUENCE

        best_pscore_id = np.argmax(pscore)
        best_pscore = pscore[best_pscore_id]

        target = delta[:, best_pscore_id] / scale_z
        target_sz = target_sz / scale_z
        lr = penalty[best_pscore_id] * score[best_pscore_id] * self.cfg.TRACK.LR

        if best_pscore >= cfg.TRACK.CONFIDENCE_LOW:
            res_x = target[0] + target_pos[0]
            res_y = target[1] + target_pos[1]
            res_w = target_sz[0] * (1 - lr) + target[2] * lr
            res_h = target_sz[1] * (1 - lr) + target[3] * lr
        else:
            res_x = target_pos[0]
            res_y = target_pos[1]
            res_w = target_sz[0]
            res_h = target_sz[1]

        if best_pscore < cfg.TRACK.CONFIDENCE_LOW:
            self.longterm_state = True
            # print("YO")
        elif best_pscore > cfg.TRACK.CONFIDENCE_HIGH:
            self.longterm_state = False

        # res_x = target[0] + target_pos[0]
        # res_y = target[1] + target_pos[1]
        # res_w = target_sz[0] * (1 - lr) + target[2] * lr
        # res_h = target_sz[1] * (1 - lr) + target[3] * lr

        target_pos = np.array([res_x, res_y])
        target_sz = np.array([res_w, res_h])
        return target_pos, target_sz, score[best_pscore_id]

    def init(self, im, bbox, GAN=None):
        self.longterm_state = False

        target_pos = np.array([bbox[0] + (bbox[2] - 1) / 2,
                               bbox[1] + (bbox[3] - 1) / 2])

        target_sz = bbox[2], bbox[3]

        state = dict()

        state['im_h'] = im.shape[0]
        state['im_w'] = im.shape[1]

        if self.longterm_state:
            instance_size = cfg.TRACK.LOST_INSTANCE_SIZE  # small object big search region
        else:
            instance_size = cfg.TRACK.INSTANCE_SIZE

        self.score_size = (instance_size - self.cfg.TRACK.EXEMPLAR_SIZE) / self.total_stride + 1

        self.anchor = self.generate_anchor(self.total_stride, self.cfg.ANCHOR.SCALES,
                                           self.cfg.ANCHOR.RATIOS, int(self.score_size))

        avg_chans = np.mean(im, axis=(0, 1))

        wc_z = target_sz[0] + self.cfg.TRACK.CONTEXT_AMOUNT * sum(target_sz)
        hc_z = target_sz[1] + self.cfg.TRACK.CONTEXT_AMOUNT * sum(target_sz)
        s_z = round(np.sqrt(wc_z * hc_z))
        # initialize the exemplar
        z_crop = get_subwindow_tracking(im, target_pos, self.cfg.TRACK.EXEMPLAR_SIZE, s_z, avg_chans)

        z = Variable(z_crop.unsqueeze(0))
        self.net.temple(z.cuda())

        if self.windowing == 'cosine':
            window = np.outer(np.hanning(self.score_size), np.hanning(self.score_size))
        elif self.windowing == 'uniform':
            window = np.ones((self.score_size, self.score_size))
        window = np.tile(window.flatten(), self.cfg.ANCHOR.ANCHOR_NUM)

        state['net'] = self.net
        state['avg_chans'] = avg_chans
        state['window'] = window
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz
        self.state = state
        self.lost = 0

    def init_adv(self, im, bbox, GAN):
        self.longterm_state = False

        target_pos = np.array([bbox[0] + (bbox[2] - 1) / 2,
                               bbox[1] + (bbox[3] - 1) / 2])

        target_sz = bbox[2], bbox[3]

        state = dict()

        state['im_h'] = im.shape[0]
        state['im_w'] = im.shape[1]

        if self.longterm_state:
            instance_size = cfg.TRACK.LOST_INSTANCE_SIZE  # small object big search region
        else:
            instance_size = cfg.TRACK.INSTANCE_SIZE

        self.score_size = (instance_size - self.cfg.TRACK.EXEMPLAR_SIZE) / self.total_stride + 1

        self.anchor = self.generate_anchor(self.total_stride, self.cfg.ANCHOR.SCALES,
                                           self.cfg.ANCHOR.RATIOS, int(self.score_size))

        avg_chans = np.mean(im, axis=(0, 1))

        wc_z = target_sz[0] + self.cfg.TRACK.CONTEXT_AMOUNT * sum(target_sz)
        hc_z = target_sz[1] + self.cfg.TRACK.CONTEXT_AMOUNT * sum(target_sz)
        s_z = round(np.sqrt(wc_z * hc_z))
        # initialize the exemplar
        z_crop = get_subwindow_tracking(im, target_pos, self.cfg.TRACK.EXEMPLAR_SIZE, s_z, avg_chans)

        z = Variable(z_crop.unsqueeze(0))

        perturbmetrics = {"metrics": {"MAE": torch.tensor(0.0), "SSIM": 100}}

        z, perturbmetrics = adv_attack_template(z.cuda(), GAN)

        self.net.temple(z.cuda())

        if self.windowing == 'cosine':
            window = np.outer(np.hanning(self.score_size), np.hanning(self.score_size))
        elif self.windowing == 'uniform':
            window = np.ones((self.score_size, self.score_size))
        window = np.tile(window.flatten(), self.cfg.ANCHOR.ANCHOR_NUM)

        state['net'] = self.net
        state['avg_chans'] = avg_chans
        state['window'] = window
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz
        state.update(perturbmetrics)
        self.state = state
        self.lost = 0

        return state

    def init_adv_T(self, im, bbox, GAN):
        self.longterm_state = False

        target_pos = np.array([bbox[0] + (bbox[2] - 1) / 2,
                               bbox[1] + (bbox[3] - 1) / 2])

        target_sz = bbox[2], bbox[3]

        state = dict()

        state['im_h'] = im.shape[0]
        state['im_w'] = im.shape[1]

        if self.longterm_state:
            instance_size = cfg.TRACK.LOST_INSTANCE_SIZE  # small object big search region
        else:
            instance_size = cfg.TRACK.INSTANCE_SIZE

        self.score_size = (instance_size - self.cfg.TRACK.EXEMPLAR_SIZE) / self.total_stride + 1

        self.anchor = self.generate_anchor(self.total_stride, self.cfg.ANCHOR.SCALES,
                                           self.cfg.ANCHOR.RATIOS, int(self.score_size))

        avg_chans = np.mean(im, axis=(0, 1))

        wc_z = target_sz[0] + self.cfg.TRACK.CONTEXT_AMOUNT * sum(target_sz)
        hc_z = target_sz[1] + self.cfg.TRACK.CONTEXT_AMOUNT * sum(target_sz)
        s_z = round(np.sqrt(wc_z * hc_z))
        # initialize the exemplar
        z_crop = get_subwindow_tracking(im, target_pos, self.cfg.TRACK.EXEMPLAR_SIZE, s_z, avg_chans)

        z = Variable(z_crop.unsqueeze(0))

        perturbmetrics = {"metrics": {"MAE": torch.tensor(0.0), "SSIM": 100}}

        # z, perturbmetrics = adv_attack_template(z.cuda(), GAN)

        self.net.temple(z.cuda())
        self.template = z.cuda()

        if self.windowing == 'cosine':
            window = np.outer(np.hanning(self.score_size), np.hanning(self.score_size))
        elif self.windowing == 'uniform':
            window = np.ones((self.score_size, self.score_size))
        window = np.tile(window.flatten(), self.cfg.ANCHOR.ANCHOR_NUM)

        state['net'] = self.net
        state['avg_chans'] = avg_chans
        state['window'] = window
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz
        state.update(perturbmetrics)
        self.state = state
        self.lost = 0
        return tensor2img(z_crop)

        # return state

    def track_advT(self, im, GAN, dir_):

        state = self.state
        net = state['net']
        avg_chans = state['avg_chans']
        window = state['window']
        target_pos = state['target_pos']
        target_sz = state['target_sz']

        wc_z = target_sz[1] + self.cfg.TRACK.CONTEXT_AMOUNT * sum(target_sz)
        hc_z = target_sz[0] + self.cfg.TRACK.CONTEXT_AMOUNT * sum(target_sz)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = self.cfg.TRACK.EXEMPLAR_SIZE / s_z

        if self.longterm_state:
            instance_size = cfg.TRACK.LOST_INSTANCE_SIZE
            self.lost += 1

        else:
            instance_size = cfg.TRACK.INSTANCE_SIZE

        d_search = (instance_size - self.cfg.TRACK.EXEMPLAR_SIZE) / 2
        pad = d_search / scale_z
        s_x = s_z + 2 * pad

        # extract scaled crops for search region x at previous target position
        x_crop = Variable(get_subwindow_tracking(
            im, target_pos, instance_size, round(s_x), avg_chans).unsqueeze(0))

        #print(x_crop.shape, self.longterm_state, cfg.TRACK.INSTANCE_SIZE)
        # exit()

        #x_crop, perturbmetrics = adv_attack_search(x_crop.cuda(), GAN, (instance_size, instance_size))
        x_crop, perturbmetrics = adv_attack_search_T(
            x_crop.cuda(), self.template, GAN, dir_, (instance_size, instance_size))

        target_sz1 = [target_sz[0] * scale_z, target_sz[1] * scale_z]

        target_pos, target_sz, score = self.tracker_eval(
            net, x_crop.cuda(), target_pos, target_sz1, window, scale_z)
        target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
        target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
        target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
        target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz
        state['best_score'] = score
        self.state = state

        state['bbox'] = [target_pos[0] - target_sz[0] / 2,
                         target_pos[1] - target_sz[1] / 2,
                         target_sz[0],
                         target_sz[1]]
        state['cropx'] = x_crop

        # metrics = {"MAE": torch.tensor(0.0), "SSIM": 100}
        # state['metrics'] = metrics
        state['lost'] = self.lost

        state.update(perturbmetrics)
        return state

    def track(self, im):
        state = self.state
        net = state['net']
        avg_chans = state['avg_chans']
        window = state['window']
        target_pos = state['target_pos']
        target_sz = state['target_sz']

        wc_z = target_sz[1] + self.cfg.TRACK.CONTEXT_AMOUNT * sum(target_sz)
        hc_z = target_sz[0] + self.cfg.TRACK.CONTEXT_AMOUNT * sum(target_sz)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = self.cfg.TRACK.EXEMPLAR_SIZE / s_z

        if self.longterm_state:
            instance_size = cfg.TRACK.LOST_INSTANCE_SIZE
            self.lost += 1

        else:
            instance_size = cfg.TRACK.INSTANCE_SIZE

        d_search = (instance_size - self.cfg.TRACK.EXEMPLAR_SIZE) / 2
        pad = d_search / scale_z
        s_x = s_z + 2 * pad

        # extract scaled crops for search region x at previous target position
        x_crop = Variable(get_subwindow_tracking(
            im, target_pos, instance_size, round(s_x), avg_chans).unsqueeze(0))

        target_sz1 = [target_sz[0] * scale_z, target_sz[1] * scale_z]

        target_pos, target_sz, score = self.tracker_eval(
            net, x_crop.cuda(), target_pos, target_sz1, window, scale_z)
        target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
        target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
        target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
        target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz
        state['best_score'] = score
        self.state = state

        state['bbox'] = [target_pos[0] - target_sz[0] / 2,
                         target_pos[1] - target_sz[1] / 2,
                         target_sz[0],
                         target_sz[1]]
        state['cropx'] = x_crop
        state['lost'] = self.lost

        # metrics = {"MAE": torch.tensor(0.0), "SSIM": 100}
        # state['metrics'] = metrics

        return state

    def init_adv_S(self, im, bbox, GAN):

        target_pos = np.array([bbox[0] + (bbox[2] - 1) / 2,
                               bbox[1] + (bbox[3] - 1) / 2])

        target_sz = bbox[2], bbox[3]

        state = dict()

        state['im_h'] = im.shape[0]
        state['im_w'] = im.shape[1]

        if self.longterm_state:
            instance_size = cfg.TRACK.LOST_INSTANCE_SIZE  # small object big search region
        else:
            instance_size = cfg.TRACK.INSTANCE_SIZE

        self.score_size = (instance_size - self.cfg.TRACK.EXEMPLAR_SIZE) / self.total_stride + 1

        self.anchor = self.generate_anchor(self.total_stride, self.cfg.ANCHOR.SCALES,
                                           self.cfg.ANCHOR.RATIOS, int(self.score_size))

        avg_chans = np.mean(im, axis=(0, 1))

        wc_z = target_sz[0] + self.cfg.TRACK.CONTEXT_AMOUNT * sum(target_sz)
        hc_z = target_sz[1] + self.cfg.TRACK.CONTEXT_AMOUNT * sum(target_sz)
        s_z = round(np.sqrt(wc_z * hc_z))
        # initialize the exemplar
        z_crop = get_subwindow_tracking(im, target_pos, self.cfg.TRACK.EXEMPLAR_SIZE, s_z, avg_chans)

        z = Variable(z_crop.unsqueeze(0))
        z = adv_attack_template_S(z.cuda(), GAN)

        self.net.temple(z.cuda())

        if self.windowing == 'cosine':
            window = np.outer(np.hanning(self.score_size), np.hanning(self.score_size))
        elif self.windowing == 'uniform':
            window = np.ones((self.score_size, self.score_size))
        window = np.tile(window.flatten(), self.cfg.ANCHOR.ANCHOR_NUM)

        state['net'] = self.net
        state['avg_chans'] = avg_chans
        state['window'] = window
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz

        self.state = state
