# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from pysot.core.config import cfg
from pysot.utils.anchor import Anchors
from pysot.tracker.base_tracker import SiameseTracker
from data_utils import tensor2img
import cv2
import os
''' Adversarial Attack tools :)'''
from attack_utils import adv_attack_template, adv_attack_search, adv_attack_search_T, \
    add_gauss_noise, add_pulse_noise, adv_attack_template_S


def crop_and_resize(img, center, size, out_size,
                    border_type=cv2.BORDER_CONSTANT,
                    border_value=(0, 0, 0),
                    interp=cv2.INTER_LINEAR):
    # convert box to corners (0-indexed)
    size = round(size)
    corners = np.concatenate((
        np.round(center - (size - 1) / 2),
        np.round(center - (size - 1) / 2) + size))
    corners = np.round(corners).astype(int)

    # pad image if necessary
    pads = np.concatenate((
        -corners[:2], corners[2:] - img.shape[:2]))
    npad = max(0, int(pads.max()))
    if npad > 0:
        img = cv2.copyMakeBorder(
            img, npad, npad, npad, npad,
            border_type, value=border_value)

    # crop image patch
    corners = (corners + npad).astype(int)
    patch = img[corners[0]:corners[2], corners[1]:corners[3]]

    # resize to out_size
    patch = cv2.resize(patch, (out_size, out_size),
                       interpolation=interp)

    return patch


class _BatchNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features, *args, **kwargs):
        super(_BatchNorm2d, self).__init__(
            num_features, *args, eps=1e-6, momentum=0.05, **kwargs)


class _AlexNet(nn.Module):

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class AlexNetV1(_AlexNet):
    output_stride = 8

    def __init__(self):
        super(AlexNetV1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            _BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, groups=2))


class SiamFC(nn.Module):

    def __init__(self, out_scale=0.001):
        super(SiamFC, self).__init__()
        self.out_scale = out_scale

    def forward(self, z, x):
        return self._fast_xcorr(z, x) * self.out_scale

    def _fast_xcorr(self, z, x):
        # fast cross correlation
        nz = z.size(0)
        nx, c, h, w = x.size()
        x = x.view(-1, nz * c, h, w)
        out = F.conv2d(x, z, groups=nz)
        out = out.view(nx, -1, out.size(-2), out.size(-1))
        return out


class Net(nn.Module):

    def __init__(self, backbone, head):
        super(Net, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, z, x):
        z = self.backbone(z)
        x = self.backbone(x)
        return self.head(z, x)


class SiamFCTracker(SiameseTracker):
    def __init__(self, model=None, dataset=None):
        super(SiamFCTracker, self).__init__()

        self.cfg = cfg

        self.net = Net(backbone=AlexNetV1(), head=SiamFC(self.cfg.TRACK.out_scale))
        self.net = self.net.cuda()
        self.net.eval()

        self.net.load_state_dict(torch.load(
            '/cvlabdata1/home/krishna/AttTracker/GIT/CSA/pysot/experiments/siamfc_alex/model.pth', map_location=lambda storage, loc: storage))

    def init(self, img, box):
        self.net.eval()

        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]

        self.upscale_sz = self.cfg.TRACK.response_up * self.cfg.TRACK.response_sz

        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()

        self.scale_factors = self.cfg.TRACK.scale_step ** np.linspace(
            -(self.cfg.TRACK.scale_num // 2),
            self.cfg.TRACK.scale_num // 2, self.cfg.TRACK.scale_num)

        # exemplar and search sizes
        context = self.cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
            self.cfg.TRACK.INSTANCE_SIZE / self.cfg.TRACK.EXEMPLAR_SIZE

        # exemplar image
        self.avg_color = np.mean(img, axis=(0, 1))
        z = crop_and_resize(
            img, self.center, self.z_sz,
            out_size=self.cfg.TRACK.EXEMPLAR_SIZE,
            border_value=self.avg_color)

        # exemplar features
        z = torch.from_numpy(z).cuda().permute(2, 0, 1).unsqueeze(0).float()
        self.kernel = self.net.backbone(z)

    def init_adv_T(self, img, box, GAN):
        self.net.eval()

        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]

        self.upscale_sz = self.cfg.TRACK.response_up * self.cfg.TRACK.response_sz

        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()

        self.scale_factors = self.cfg.TRACK.scale_step ** np.linspace(
            -(self.cfg.TRACK.scale_num // 2),
            self.cfg.TRACK.scale_num // 2, self.cfg.TRACK.scale_num)

        # exemplar and search sizes
        context = self.cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
            self.cfg.TRACK.INSTANCE_SIZE / self.cfg.TRACK.EXEMPLAR_SIZE

        # exemplar image
        self.avg_color = np.mean(img, axis=(0, 1))
        z = crop_and_resize(
            img, self.center, self.z_sz,
            out_size=self.cfg.TRACK.EXEMPLAR_SIZE,
            border_value=self.avg_color)

        # exemplar features
        z = torch.from_numpy(z).cuda().permute(2, 0, 1).unsqueeze(0).float()

        self.template = z

        # z, perturbmetrics = adv_attack_template(z, GAN)

        # output_dict = {}
        # output_dict.update(perturbmetrics)

        self.kernel = self.net.backbone(z)

        return tensor2img(z)

    @torch.no_grad()
    def update(self, img, GAN):
        self.net.eval()

        # search images
        x = [crop_and_resize(
            img, self.center, self.x_sz * f,
            out_size=self.cfg.TRACK.INSTANCE_SIZE,
            border_value=self.avg_color) for f in self.scale_factors]
        x = np.stack(x, axis=0)
        x = torch.from_numpy(x).cuda().permute(0, 3, 1, 2).float()

        x, perturbmetrics = adv_attack_search(x, GAN)

        output_dict = {}
        output_dict['cropx'] = x

        # responses
        x = self.net.backbone(x)
        responses = self.net.head(self.kernel, x)
        responses = responses.squeeze(1).cpu().numpy()

        # upsample responses and penalize scale changes
        responses = np.stack([cv2.resize(
            u, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC)
            for u in responses])
        responses[: self.cfg.TRACK.scale_num // 2] *= self.cfg.TRACK.PENALTY_K
        responses[self.cfg.TRACK.scale_num // 2 + 1:] *= self.cfg.TRACK.PENALTY_K

        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        response = responses[scale_id]
        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - self.cfg.TRACK.WINDOW_INFLUENCE) * response + \
            self.cfg.TRACK.WINDOW_INFLUENCE * self.hann_window
        loc = np.unravel_index(response.argmax(), response.shape)

        # locate target center
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        disp_in_instance = disp_in_response * \
            self.cfg.TRACK.total_stride / self.cfg.TRACK.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
            self.scale_factors[scale_id] / self.cfg.TRACK.INSTANCE_SIZE
        self.center += disp_in_image

        # update target size
        scale = (1 - self.cfg.TRACK.LR) * 1.0 + \
            self.cfg.TRACK.LR * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        output_dict.update({'bbox': box, 'best_score': 1.0})
        output_dict.update(perturbmetrics)

        return output_dict

    def track_adv(self, img, GAN):
        output_dict = self.update(img, GAN)
        # metrics = {"MAE": torch.tensor(0.0), "SSIM": 100}
        # output_dict.update({"metrics": metrics})

        # # print("YO")
        return output_dict

    @torch.no_grad()
    def track_advT(self, img, GAN, dir_):

        self.net.eval()

        # search images
        x = [crop_and_resize(
            img, self.center, self.x_sz * f,
            out_size=self.cfg.TRACK.INSTANCE_SIZE,
            border_value=self.avg_color) for f in self.scale_factors]
        x = np.stack(x, axis=0)
        x = torch.from_numpy(x).cuda().permute(0, 3, 1, 2).float()

        #x, perturbmetrics = adv_attack_search(x.cuda(), GAN)
        with torch.no_grad():
            x, perturbmetrics = adv_attack_search_T(x, self.template, GAN, dir_)

        #perturbmetrics = {"metrics": {"MAE": torch.tensor(0.0), "SSIM": 100}}
        output_dict = {}
        #output_dict['cropx'] = x.detach().cpu().numpy()
        output_dict['cropx'] = 0

        # print(x.shape)

        # responses
        x = self.net.backbone(x)
        responses = self.net.head(self.kernel, x)
        responses = responses.squeeze(1).detach().cpu().numpy()

        # upsample responses and penalize scale changes
        responses = np.stack([cv2.resize(
            u, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC)
            for u in responses])
        responses[: self.cfg.TRACK.scale_num // 2] *= self.cfg.TRACK.PENALTY_K
        responses[self.cfg.TRACK.scale_num // 2 + 1:] *= self.cfg.TRACK.PENALTY_K

        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        response = responses[scale_id]
        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - self.cfg.TRACK.WINDOW_INFLUENCE) * response + \
            self.cfg.TRACK.WINDOW_INFLUENCE * self.hann_window
        loc = np.unravel_index(response.argmax(), response.shape)

        # print(loc)

        # locate target center
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        disp_in_instance = disp_in_response * \
            self.cfg.TRACK.total_stride / self.cfg.TRACK.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
            self.scale_factors[scale_id] / self.cfg.TRACK.INSTANCE_SIZE
        self.center += disp_in_image

        # update target size
        scale = (1 - self.cfg.TRACK.LR) * 1.0 + \
            self.cfg.TRACK.LR * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        output_dict['bbox'] = box
        output_dict['best_score'] = response[loc]

        output_dict.update(perturbmetrics)
        return output_dict
