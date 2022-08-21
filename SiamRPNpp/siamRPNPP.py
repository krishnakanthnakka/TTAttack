from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import cv2
import torch
import os
import torch.nn.functional as F

from common_path import project_path_
from pysot.models.model_builder import ModelBuilder
from pysot.utils.model_load import load_pretrain
from pysot.core.config import cfg
from pysot.utils.anchor import Anchors
from pysot.utils.bbox import IoU, corner2center, center2corner
from pysot.models.loss import (select_cross_entropy_loss, select_cross_entropy_loss_pos,
                               weight_l1_loss)
from pysot.tracker.tracker_builder import build_tracker
from torchvision.utils import save_image
import os
root_dir = os.path.dirname(os.path.abspath(__file__))
vis_dir = os.path.join(root_dir, "vis_dir")


class SiamRPNPP():
    def __init__(self, dataset='', tracker_name='', istrain=False):

        # hanning = np.hanning(17)
        # hanning = np.outer(hanning, hanning)
        # np.savetxt("hanning17.txt", hanning, fmt="%.2f")
        # exit()
        """
        if 'OTB' in dataset:
            cfg_file = os.path.join(
                project_path_, 'pysot/experiments/siamrpn_r50_l234_dwxcorr_otb/config.yaml')
            snapshot = os.path.join(
                project_path_, 'pysot/experiments/siamrpn_r50_l234_dwxcorr_otb/model.pth')

        elif 'LT' in dataset:
            cfg_file = os.path.join(
                project_path_, 'pysot/experiments/siamrpn_r50_l234_dwxcorr_lt/config.yaml')
            snapshot = os.path.join(
                project_path_, 'pysot/experiments/siamrpn_r50_l234_dwxcorr_lt/model.pth')
        else:

            # -------------- MODIFIED BELOW

            # Shallow
            # cfg_file = os.path.join(project_path_, 'pysot/experiments/siamrpn_r50_l234_dwxcorr/config.yaml')
            # snapshot = os.path.join(project_path_, 'pysot/experiments/siamrpn_r50_l234_dwxcorr/model.pth')

            # DEEP
            # cfg_file = os.path.join(project_path_, 'pysot/experiments/siamrpn_r50_l234_dwxcorr_lt/config.yaml')
            # snapshot = os.path.join(project_path_, 'pysot/experiments/siamrpn_r50_l234_dwxcorr_lt/model.pth')

            # SiamRPN
            # cfg_file = os.path.join(project_path_, 'pysot/experiments/siamrpn_alex_dwxcorr/config.yaml')
            # snapshot = os.path.join(project_path_, 'pysot/experiments/siamrpn_alex_dwxcorr/model.pth')

            # DAASiamRPN
            # cfg_file = os.path.join(project_path_, 'pysot/experiments/DAsiamrpn_alex_dwxcorr/config.yaml')
            # snapshot = os.path.join(project_path_, 'pysot/experiments/DAsiamrpn_alex_dwxcorr/modelVOT.pth')

        """

        #  TODO, Cheeck this, it creaates error for targeted attacks on others

        # print("YO", tracker_name)

        # exit()

        if not istrain:

            ''' before
            cfg_file = os.path.join(
                project_path_, 'pysot/experiments/siamrpn_r50_l234_dwxcorr/config.yaml')
            snapshot = os.path.join(
                project_path_, 'pysot/experiments/siamrpn_r50_l234_dwxcorr/model.pth')
            '''

            if tracker_name in ['siamrpn_r50_l234_dwxcorr', 'siamrpn_r50_l234_dwxcorr_otb']:
                cfg_file = os.path.join(
                    project_path_, 'pysot/experiments/siamrpn_r50_l234_dwxcorr/config.yaml')
                snapshot = os.path.join(
                    project_path_, 'pysot/experiments/siamrpn_r50_l234_dwxcorr/model.pth')

            elif tracker_name in ['siamrpn_r50_l234_dwxcorr_lt']:
                cfg_file = os.path.join(
                    project_path_, 'pysot/experiments/siamrpn_r50_l234_dwxcorr_lt/config.yaml')
                snapshot = os.path.join(
                    project_path_, 'pysot/experiments/siamrpn_r50_l234_dwxcorr_lt/model.pth')

            elif tracker_name in ['siamrpn_alex_dwxcorr', 'siamrpn_alex_dwxcorr_otb', 'DAsiamrpn_alex_dwxcorr']:
                cfg_file = os.path.join(
                    project_path_, 'pysot/experiments/siamrpn_alex_dwxcorr/config.yaml')
                snapshot = os.path.join(
                    project_path_, 'pysot/experiments/siamrpn_alex_dwxcorr/model.pth')

            elif tracker_name in ['siamrpn_mobilev2_l234_dwxcorr']:
                cfg_file = os.path.join(
                    project_path_, 'pysot/experiments/siamrpn_mobilev2_l234_dwxcorr/config.yaml')
                snapshot = os.path.join(
                    project_path_, 'pysot/experiments/siamrpn_mobilev2_l234_dwxcorr/model.pth')

            elif tracker_name in ['dimp']:
                return

            elif tracker_name in ['siam_ocean', 'siam_ocean_online']:
                print("RETURNING FROM SIAMRPN++ FILE")

                return

            elif tracker_name in ['siamrpn_ban_r50_l234_otb']:
                cfg_file = os.path.join(
                    project_path_, 'pysot/experiments/siamrpn_ban_r50_l234_otb/config.yaml')
                snapshot = os.path.join(
                    project_path_, 'pysot/experiments/siamrpn_ban_r50_l234_otb/model.pth')

            else:

                assert False, " No tracker found at training ttime"

            cfg.merge_from_file(cfg_file)

            # CHANGED
            # self.model = ModelBuilder()  # A Neural Network.(a torch.nn.Module)
            # self.model = load_pretrain(self.model, snapshot).cuda().eval()

        else:
            if tracker_name in ['siamrpn_r50_l234_dwxcorr', 'siamrpn_r50_l234_dwxcorr_otb']:
                cfg_file = os.path.join(
                    project_path_, 'pysot/experiments/siamrpn_r50_l234_dwxcorr/config.yaml')
                snapshot = os.path.join(
                    project_path_, 'pysot/experiments/siamrpn_r50_l234_dwxcorr/model.pth')

            elif tracker_name in ['siamrpn_r50_l234_dwxcorr_lt']:
                cfg_file = os.path.join(
                    project_path_, 'pysot/experiments/siamrpn_r50_l234_dwxcorr_lt/config.yaml')
                snapshot = os.path.join(
                    project_path_, 'pysot/experiments/siamrpn_r50_l234_dwxcorr_lt/model.pth')

            elif tracker_name in ['siamrpn_alex_dwxcorr', 'DAsiamrpn_alex_dwxcorr']:
                cfg_file = os.path.join(
                    project_path_, 'pysot/experiments/siamrpn_alex_dwxcorr/config.yaml')
                snapshot = os.path.join(
                    project_path_, 'pysot/experiments/siamrpn_alex_dwxcorr/model.pth')

            else:

                assert False, " No tracker found at training ttime"

                # -------------------- MODIFIED BELOW --------------------

                # Shallow
                # cfg_file = os.path.join(project_path_, 'pysot/experiments/siamrpn_r50_l234_dwxcorr/config.yaml')
                # snapshot = os.path.join(project_path_, 'pysot/experiments/siamrpn_r50_l234_dwxcorr/model.pth')

                # DEEP
                # cfg_file = os.path.join(project_path_, 'pysot/experiments/siamrpn_r50_l234_dwxcorr_lt/config.yaml')
                # snapshot = os.path.join(project_path_, 'pysot/experiments/siamrpn_r50_l234_dwxcorr_lt/model.pth')

                # SiamRPN
                # cfg_file = os.path.join(project_path_, 'pysot/experiments/siamrpn_alex_dwxcorr/config.yaml')
                # snapshot = os.path.join(project_path_, 'pysot/experiments/siamrpn_alex_dwxcorr/model.pth')

                # DAASiamRPN
                # cfg_file = os.path.join(project_path_, 'pysot/experiments/DAsiamrpn_alex_dwxcorr/config.yaml')
                # snapshot = os.path.join(project_path_, 'pysot/experiments/DAsiamrpn_alex_dwxcorr/modelVOT.pth')

            # print(cfg_file, "KK")
            # exit()
            cfg.merge_from_file(cfg_file)

            self.model = ModelBuilder()  # A Neural Network.(a torch.nn.Module)
            self.model = load_pretrain(self.model, snapshot).cuda().eval()

       # self.model = build_tracker(model=None, dataset='OTB100').net

        #self.target_feats = torch.load(os.path.join(root_dir, 'data/target.pth'))

        self.anchors = self.generate_anchor()
        self.singleanchors = np.zeros((4, 5))

        self.singleanchors[0] = [0, 0, 0, 0, 0]
        self.singleanchors[1] = [0, 0, 0, 0, 0]
        self.singleanchors[2] = [104, 88, 64, 40, 32]
        self.singleanchors[3] = [32, 40, 64, 80, 96]

        self.maxratios = []
        # print(cfg, istrain, cfg_file, tracker_name)

    def get_heat_map(self, x_crop, softmax=False, index=0):
        outputs = self.model.track(x_crop)
        score_map = outputs['cls']  # (N,2x5,25,25)

        # print(score_map.shape)

        score_map = score_map.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)  # (5HWN,2)
        if softmax:
            score_map = F.softmax(score_map, dim=1).data[:, 1]  # (5HWN,)

        anchorindex = []

        if True:
            score = self._convert_score(outputs['cls'])
            pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

            # x_crop_all = x_crop.clone()

            # for j in range(x_crop.shape[0]):
            #     best_idx = np.argmax(score[j])
            #     best_score = np.max(score[j])
            #     bbox_search = pred_bbox[j, :, best_idx]
            #     print([0, 0, int(bbox_search[2]), int(bbox_search[3])])
            #     ratios = IoU(self.singleanchors, [0, 0, int(bbox_search[2]), int(bbox_search[3])])
            #     print(np.argmax(ratios))
            #     anchorindex.append(np.argmax(ratios))

            #     x_crop_j = x_crop[j].detach().cpu().numpy()
            #     x_crop_j = np.ascontiguousarray(x_crop_j.transpose(1, 2, 0))
            #     x_crop_j = x_crop_j.astype('uint8')
            #     offset = cfg.TRACK.INSTANCE_SIZE / 2
            #     cv2.rectangle(x_crop_j, (int(offset + bbox_search[0] - bbox_search[2] / 2),
            # int(offset + bbox_search[1] -
            #     bbox_search[3] / 2)), (int(offset + bbox_search[0] + bbox_search[2] / 2), int(offset +
            #      bbox_search[1] + bbox_search[3] / 2)), (128, 128, 0), 3)
            #     cv2.putText(x_crop_j, str("{:.2f}".format(best_score)), (int(
            #         offset + bbox_search[0] - bbox_search[2] / 2), int(offset + bbox_search[1] - bbox_search[3] / 2) - 10), 0, 0.4, 255)
            #     # cv2.imwrite(os.path.join(vis_dir, "temp_{}.png".format(j)), x_crop_j)
            #     x_crop_all[j] = torch.tensor(x_crop_j.transpose(2, 0, 1))
            # save_image((x_crop_all) / 255, os.path.join(vis_dir,
            #                                             "{}_real.png".format(index)), nrow=4)
            # print(anchorindex)

            best_idx = np.argmax(score, 1)
            rowinds = torch.arange(score.shape[0])
            bbox_search = pred_bbox[rowinds, :, best_idx]
            bbox_search[:, :2] = 0
            ratios = [IoU(self.singleanchors, search_pred) for search_pred in bbox_search]
            maxratios = np.argmax(ratios, 1)
            self.maxratios = maxratios

        return score_map

    def get_cls_reg(self, X_crop, softmax=False):
        outputs = self.model.track(X_crop)  # (N,2x5,25,25)
        score_map = outputs['cls'].permute(1, 2, 3, 0).contiguous().view(
            2, -1).permute(1, 0)  # (5HWN,2)
        reg_res = outputs['loc'].permute(1, 2, 3, 0).contiguous().view(4, -1)
        if softmax:
            score_map = F.softmax(score_map, dim=1).data[:, 1]  # (5HWN,)

        self.outputs = outputs

        return score_map, reg_res

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2 // 2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def generate_all_anchors(self):

        anchors2 = Anchors(cfg.ANCHOR.STRIDE, cfg.ANCHOR.RATIOS, cfg.ANCHOR.SCALES)
        anchors2.generate_all_anchors(im_c=0, size=cfg.TRACK.OUTPUT_SIZE)
        # anchors2.generate_all_anchors(im_c=0, size=cfg.TRAIN.OUTPUT_SIZE)

        return anchors2

    def get_target_cls_reg(self, X_crop, shift, centerpoint):

        output_dict = {}
        outputs = self.outputs  # (N,2x5,25,25)
        pred_cls = outputs['cls']
        pred_reg = outputs['loc']
        b, a2, h, w = pred_cls.size()

        assert centerpoint[0] == (h - 1) / 2
        assert centerpoint[1] == (w - 1) / 2
        assert cfg.TRACK.OUTPUT_SIZE == h

        anchor_num = a2 // 2
        anchors2 = self.generate_all_anchors()
        anchor_box = anchors2.all_anchors[0]
        anchor_center = anchors2.all_anchors[1]

        assert anchor_box.shape[2] == h, 'Wrong anchor location'

        label_cls = 0 * np.ones((b, anchor_num, h, w), dtype=np.int64)
        delta = np.zeros((b, 4, anchor_num, h, w), dtype=np.float32)
        delta_weight = np.zeros((b, anchor_num, h, w), dtype=np.float32)
        label_cls_weight = 0 * np.ones((b, anchor_num, h, w), dtype=np.int64)

        # -----   CHANGE WHILE TRAINING

        pos = [2, centerpoint[0] + shift[1], centerpoint[1] + shift[0]]
        target = [int(anchor_box[i, pos[0], pos[1], pos[2]]) for i in range(4)]

        def select(position, keep_num=16):
            num = position[0].shape[0]
            if num <= keep_num:
                return position, num
            slt = np.arange(num)
            np.random.shuffle(slt)
            slt = slt[:keep_num]
            return tuple(p[slt] for p in position), keep_num

        tcx, tcy, tw, th = corner2center(target)

        x1, y1, x2, y2 = anchor_box[0], anchor_box[1], anchor_box[2], anchor_box[3]
        cx, cy, w, h = anchor_center[0], anchor_center[1], anchor_center[2], anchor_center[3]
        delta[:, 0] = (tcx - cx) / w
        delta[:, 1] = (tcy - cy) / h
        delta[:, 2] = np.log(tw / w)
        delta[:, 3] = np.log(th / h)

        bbox_target = [int(anchor_box[i, pos[0], pos[1], pos[2]]) for i in range(4)]
        output_dict['bbox_target'] = bbox_target

        if 0:
            print("Adv Target : ", output_dict['bbox_target'])
            print(IoU(target, bbox_target))
            x_crop_img = X_crop[0].detach().cpu().numpy()
            x_crop_img = np.ascontiguousarray(x_crop_img.transpose(1, 2, 0)).astype('uint8')
            offset = cfg.TRACK.INSTANCE_SIZE / 2
            cv2.rectangle(x_crop_img, (int(offset + bbox_target[0]), int(offset + bbox_target[1])),
                          (int(offset + bbox_target[2]), int(offset + bbox_target[3])), (0, 255, 0), 3)

            bbox1 = [int(anchor_box[i, pos[0], centerpoint[0], centerpoint[1]]) for i in range(4)]
            cv2.rectangle(x_crop_img, (int(offset + bbox1[0]), int(offset + bbox1[1])),
                          (int(offset + bbox1[2]), int(offset + bbox1[3])), (0, 0, 255), 3)

            cv2.rectangle(x_crop_img, (int(offset + target[0]), int(offset + target[1])),
                          (int(offset + target[2]), int(offset + target[3])), (255, 0, 0), 3)

            cv2.imwrite("search_target_clean.png", x_crop_img)
            exit()

        label_cls[:, pos[0], pos[1], pos[2]] = 1.0
        delta_weight[:, pos[0], pos[1], pos[2]] = 1.0
        label_cls = torch.from_numpy(label_cls).cuda()
        label_loc = torch.from_numpy(delta).cuda()
        label_loc_weight = torch.from_numpy(delta_weight).cuda()

        pred_cls = self.log_softmax(pred_cls)
        cls_loss = select_cross_entropy_loss_pos(pred_cls, label_cls)
        loc_loss = weight_l1_loss(pred_reg, label_loc, label_loc_weight)

        outputs = {}
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss

        return cls_loss, loc_loss

    def get_feat_loss(self):
        pred_feats = self.outputs['xf']
        loss_feat = 0.0
        criterionL2 = torch.nn.MSELoss()

        for i, (p, t) in enumerate(zip(pred_feats, self.target_feats)):
            loss_feat += criterionL2(p, t.cuda())

        return loss_feat

    def generate_anchor(self):
        anchors = Anchors(cfg.ANCHOR.STRIDE,
                          cfg.ANCHOR.RATIOS,
                          cfg.ANCHOR.SCALES)
        anchor = anchors.anchors

        score_size = (cfg.TRACK.INSTANCE_SIZE -
                      cfg.TRACK.EXEMPLAR_SIZE) // cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)

        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), np.tile(
            yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def visualize(self, x_crop, index=0):

        outputs = self.model.track(x_crop)
        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

        x_crop_all = x_crop.clone()

        for j in range(x_crop.shape[0]):
            best_idx = np.argmax(score[j])
            best_score = np.max(score[j])

            bbox_search = pred_bbox[j, :, best_idx]
            x_crop_j = x_crop[j].detach().cpu().numpy()
            x_crop_j = np.ascontiguousarray(x_crop_j.transpose(1, 2, 0))
            x_crop_j = x_crop_j.astype('uint8')
            offset = cfg.TRACK.INSTANCE_SIZE / 2
            cv2.rectangle(x_crop_j,
                          (int(offset + bbox_search[0] - bbox_search[2] / 2),
                           int(offset + bbox_search[1] - bbox_search[3] / 2)),
                          (int(offset + bbox_search[0] + bbox_search[2] / 2),
                           int(offset + bbox_search[1] + bbox_search[3] / 2)),
                          (128, 128, 0), 3)

            cv2.putText(x_crop_j, str("{:.2f}".format(best_score)),
                        (int(offset + bbox_search[0] - bbox_search[2] / 2),
                         int(offset + bbox_search[1] - bbox_search[3] / 2) - 10),
                        0, 0.4, 255)

            # cv2.imwrite(os.path.join(vis_dir, "temp_{}.png".format(j)), x_crop_j)
            x_crop_all[j] = torch.tensor(x_crop_j.transpose(2, 0, 1))

        save_image((x_crop_all) / 255, os.path.join(vis_dir,
                                                    "vis_all_{}.png".format(index)), nrow=4)

        # exit()

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def _convert_bbox(self, delta, anchor):

        b, na, h, w = delta.shape
        delta = delta.view(b, 4, int(na / 4), h, w)
        delta = delta.data.cpu().numpy()
        delta = delta.reshape(b, 4, -1)
        delta[:, 0, :] = delta[:, 0, :] * anchor[:, 2] + anchor[:, 0]
        delta[:, 1, :] = delta[:, 1, :] * anchor[:, 3] + anchor[:, 1]
        delta[:, 2, :] = np.exp(delta[:, 2, :]) * anchor[:, 2]
        delta[:, 3, :] = np.exp(delta[:, 3, :]) * anchor[:, 3]
        return delta

    def _convert_score(self, score):

        b, na, h, w = score.shape
        score = score.view(b, 2, -1)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score
