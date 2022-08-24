# Copyright (c) SenseTime. All Rights Reserved.

"""

python test_attack_ours_target.py  --dataset OTB100  --snapshot ../../tracker_weights/siamcar_general/model_general.pth   --model_iter=4_net_G.pth --case=2 --eps=16 --attack_universal --trajcase=SE

"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import math
import cv2
import torch
import numpy as np
import math
import sys
import shutil
sys.path.append('../')

from pysotcar.core.config import cfg
from pysotcar.tracker.siamcar_tracker import SiamCARTracker
from pysotcar.utils.bbox import get_axis_aligned_bbox
from pysotcar.utils.model_load import load_pretrain
from pysotcar.models.model_builder import ModelBuilder
from toolkit.utils.region import vot_overlap, vot_float2str

from toolkit.datasets import DatasetFactory

parser = argparse.ArgumentParser(description='siamcar tracking')
parser.add_argument('--video', default='', type=str, help='eval one special video')
parser.add_argument('--dataset', type=str, default='UAV123', help='datasets')  # OTB50 LaSOT UAV123 GOT-10k
parser.add_argument('--vis', action='store_true', default=False, help='whether visualzie result')
parser.add_argument('--snapshot', type=str, default='snapshot/checkpoint_e20.pth',
                    help='snapshot of models to eval')
parser.add_argument('--config', type=str,
                    default='../experiments/siamcar_r50/config.yaml', help='config file')
parser.add_argument('--case', type=int, required=True)
parser.add_argument('--model_iter', type=str, required=True)
parser.add_argument('--eps', type=int, required=True)
parser.add_argument('--attack_universal', default=False, action='store_true',
                    help='whether visualzie result')
parser.add_argument('--directions', type=int, default=12)
parser.add_argument('--trajcase', type=str, required=True)
parser.add_argument('--targetcase', type=int)
args = parser.parse_args()

torch.set_num_threads(1)


def get_direction(cur_gt_bbox_, prev_gt_box_, idx):

    size_gt = min(cur_gt_bbox_[2], cur_gt_bbox_[3])
    size_pr = max(prev_gt_box_[2], prev_gt_box_[3])

    if abs(size_pr - size_gt) > 20 or size_pr < 20:
        enhance = True
    else:
        enhance = False

    cur_gt_bbox_ = [cur_gt_bbox_[
        0] + (cur_gt_bbox_[2] / 2), cur_gt_bbox_[1] + (cur_gt_bbox_[3] / 2)]
    prev_gt_box_ = [prev_gt_box_[
        0] + (prev_gt_box_[2] / 2), prev_gt_box_[1] + (prev_gt_box_[3] / 2)]

    x = cur_gt_bbox_[0] - prev_gt_box_[0]
    y = cur_gt_bbox_[1] - prev_gt_box_[1]
    dir_ = (math.atan2(-y, x) + 2 * math.pi) % (2 * math.pi)

    return dir_, enhance


ckpt_root_dir = '../../SiamRPNpp'


import os
root_dir = os.path.dirname(os.path.abspath(__file__))


def load_generator():

    # OURS
    if 1:

        attack_method = 'TTA'
        import sys
        sys.path.insert(0, ckpt_root_dir + '/pix2pix')

        from options.test_options0 import TestOptions
        from models import create_model
        import os

        opt = TestOptions().parse()
        # opt.tracker_name = "siamrpn_r50_l234_dwxcorr"  # CHANGED from dimp

        opt.tracker_name = "siamrpn_mobilev2_l234_dwxcorr"  # HACK. CHANGED FROM dimp

        opt.istargeted = True
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

        opt.model = 'G_template_L2_500_regress'
        opt.netG = 'unet_128'

        #expcase, model_epoch,  opt.eps = 54, '4_net_G.pth', 8
        #expcase, model_epoch, opt.eps = 22, '8_net_G.pth', 8
        expcase, model_epoch, opt.eps, opt.directions = args.case, args.model_iter, args.eps, args.directions

        ckpt = os.path.join(ckpt_root_dir,
                            'checkpoints/{}_{}/{}'.format(opt.model, expcase, model_epoch))
        print(" Loading generator trained with OURS approach")

    GAN = create_model(opt)
    print("ckpt path:", ckpt)
    GAN.load_path = ckpt
    GAN.setup(opt)
    GAN.eval()
    return GAN, expcase, attack_method


GAN, expcase, attack_method = load_generator()


def main():
    # load config

    args.targetcase = args.trajcase

    cfg.merge_from_file(args.config)

    # hp_search
    params = getattr(cfg.HP_SEARCH, args.dataset)
    hp = {'lr': params[0], 'penalty_k': params[1], 'window_lr': params[2]}

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(cur_dir, '../../testing_dataset', args.dataset)

    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    tracker = SiamCARTracker(model, cfg.TRACK)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    model_name = args.snapshot.split('/')[-1]
    total_lost = 0

    mean_FPS = []

    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
        # restart tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            frame_counter = 0
            lost_number = 0
            toc = 0
            pred_bboxes = []
            target_bbox = []
            pred_bboxes2 = []
            target_bboxes = []
            target_bboxes_rect = []
            traj_file = os.path.join("/cvlabdata1/home/krishna/AttTracker/pysot/tools/results_paper/{}/G_template_L2_500_regress_siamrpn_r50_l234_dwxcorr/baseline/133/4_net_G.pth/{}/{}".
                                     format(args.dataset, args.targetcase, video.name), video.name + '_001_target.txt')
            with open(traj_file, 'r') as f:
                target_traj = [list(map(float, x.strip().split(','))) for x in f.readlines()]
            target_traj = np.array(target_traj)
            for idx, (img, gt_bbox) in enumerate(video):
                target_bbox = target_traj[idx]

                cx, cy, w, h = get_axis_aligned_bbox(np.array(target_bbox))
                target_bbox_rect = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]

                target_bboxes.append(target_bbox)
                target_bboxes_rect.append(target_bbox_rect)

                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                               gt_bbox[0], gt_bbox[1] + gt_bbox[3] - 1,
                               gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1] + gt_bbox[3] - 1,
                               gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1]]
                tic = cv2.getTickCount()
                if idx == frame_counter:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                    tracker.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    pred_bboxes.append(1)
                    pred_bboxes2.append(gt_bbox_)
                    prev_predbbox = target_bbox_rect

                    if idx == 0 and args.vis:
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        w, h = img.shape[:2]
                        video_out = cv2.VideoWriter(os.path.join(
                            "./viz/", video.name + ".avi"), fourcc, fps=20, frameSize=(h, w))

                elif idx > frame_counter:

                    direction = get_direction(target_bbox_rect, prev_predbbox, idx)[0]
                    outputs = tracker.track_advT(img, hp, GAN, direction, frame_id=idx)

                    pred_bbox = outputs['bbox']
                    pred_bboxes.append(pred_bbox)
                    prev_predbbox = pred_bbox
                    pred_bboxes2.append(pred_bbox)

                toc += cv2.getTickCount() - tic

                if args.vis and idx > frame_counter:
                    cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 1, 2))],
                                  True, (0, 255, 0), 3)
                    bbox = list(map(int, pred_bbox))
                    target_bbox_rect = list(map(int, target_bbox_rect))

                    cv2.rectangle(img, (bbox[0], bbox[1]),
                                  (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(img, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.rectangle(img, (target_bbox_rect[0], target_bbox_rect[1]), (
                        target_bbox_rect[0] + target_bbox_rect[2], target_bbox_rect[1] + target_bbox_rect[3]), (0, 0, 255), 3)

                    video_out.write(img)
            toc /= cv2.getTickFrequency()

            if args.vis:
                video_out.release()

            if args.attack_universal:
                video_path = os.path.join('results_U_{}_{}'.format(attack_method, expcase), args.dataset, str(
                    args.targetcase), model_name, 'baseline', video.name)
            else:
                video_path = os.path.join('results_{}_{}'.format(attack_method, expcase), args.dataset, str(
                    args.targetcase), model_name, 'baseline', video.name)

            # save results
           # video_path = os.path.join('results_{}_{}'.format(attack_method, expcase), args.dataset, model_name,
             #       'baseline', video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([vot_float2str("%.4f", i) for i in x]) + '\n')

            result_path = os.path.join(
                video_path, '{}_001_pred.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes2:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([vot_float2str("%.4f", i) for i in x]) + '\n')

            result_path = os.path.join(
                video_path, '{}_001_target.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in target_bboxes:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([vot_float2str("%.4f", i) for i in x]) + '\n')

            result_path = os.path.join(
                video_path, '{}_001_target2.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in target_bboxes_rect:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([vot_float2str("%.4f", i) for i in x]) + '\n')

            print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                v_idx + 1, video.name, toc, idx / toc, lost_number))
            total_lost += lost_number
        print("{:s} total lost: {:d}".format(model_name, total_lost))

    else:

        # OPE tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            toc = 0
            pred_bboxes = []
            track_times = []
            target_bboxes = []

            # if args.dataset in ['OTB100']:
            #     traj_file = os.path.join("/cvlabdata1/home/krishna/AttTracker/pysot/tools/results_paper/{}/G_template_L2_500_regress_siamrpn_r50_l234_dwxcorr/133/4_net_G.pth/{}/".
            #                              format(args.dataset, args.targetcase), video.name + '_target.txt')

            # elif args.dataset in ['VOT2018-LT']:
            #     traj_file = os.path.join("/cvlabdata1/home/krishna/AttTracker/pysot/tools/results/{}/G_template_L2_500_regress_siamrpn_r50_l234_dwxcorr_lt/longterm/133/4_net_G.pth/{}/{}/".
            #                              format(args.dataset, args.targetcase, video.name), video.name + '_001_target.txt')

            # elif args.dataset in ['UAV123']:
            #     traj_file = os.path.join("/cvlabdata1/home/krishna/AttTracker/pysot/tools/results_paper/{}/G_template_L2_500_regress_siamrpn_r50_l234_dwxcorr2/133/4_net_G.pth/{}/".
            #                              format(args.dataset, args.targetcase), video.name + '_target.txt')

            # elif args.dataset in ['LaSOT']:
            #     traj_file = os.path.join("/cvlabdata1/home/krishna/AttTracker/pysot/tools/results_target/{}/G_template_L2_500_regress_siamrpn_r50_l234_dwxcorr/133/4_net_G.pth/{}/".
            #                              format('lasot', args.targetcase), video.name + '_target.txt')

            if args.dataset in ['OTB100']:
                traj_file = os.path.join(root_dir, "../../", "targeted_attacks_GT", "{}/{}/".
                                         format(args.dataset, args.targetcase), video.name + '_target.txt')

            elif args.dataset in ['UAV123']:

                traj_file = os.path.join(root_dir, "../../", "targeted_attacks_GT", "{}/{}/".
                                         format(args.dataset, args.targetcase), video.name + '_target.txt')

                # traj_file = os.path.join("/cvlabdata1/home/krishna/AttTracker/pysot/tools/results_paper/{}/G_template_L2_500_regress_siamrpn_r50_l234_dwxcorr2/133/4_net_G.pth/{}/".
                #                          format(args.dataset, args.targetcase), video.name + '_target.txt')

            elif args.dataset in ['lasot']:

                traj_file = os.path.join(root_dir, "../../", "targeted_attacks_GT", "{}/{}/".
                                         format(args.dataset, args.targetcase), video.name + '_target.txt')

            with open(traj_file, 'r') as f:
                target_traj = [list(map(float, x.strip().split(','))) for x in f.readlines()]

            target_traj = np.array(target_traj)
            for idx, (img, gt_bbox) in enumerate(video):
                target_bbox = target_traj[idx]

                tic = cv2.getTickCount()
                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                    tracker.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    pred_bboxes.append(pred_bbox)
                    target_bboxes.append(pred_bbox)

                    prev_predbbox = target_bbox

                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    w, h = img.shape[:2]
                    if args.vis:
                        video_out = cv2.VideoWriter(os.path.join(
                            "./viz/", video.name + ".avi"), fourcc, fps=20, frameSize=(h, w))
                else:

                    direction, enhance = get_direction(target_bbox, prev_predbbox, idx)
                    outputs = tracker.track_advT(img, hp, GAN, direction, frame_id=idx)
                    target_bboxes.append(target_bbox)

                    pred_bbox = outputs['bbox']
                    pred_bboxes.append(pred_bbox)
                    prev_predbbox = pred_bbox

                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())

                # if idx == 0:
                #     cv2.destroyAllWindows()
                if args.vis and idx > 0:
                    if not any(map(math.isnan, gt_bbox)):
                        gt_bbox = list(map(int, gt_bbox))
                        target_bbox = list(map(int, target_bbox))

                        thickness = 2
                        extra_th = 5

                        pred_bbox = list(map(int, pred_bbox))
                        cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                      (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]), (0, 255, 0), 3)
                        cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                      (pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3]), (0, 255, 255), 3)
                        cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        cv2.rectangle(img, (target_bbox[0], target_bbox[1]), (target_bbox[0] +
                                                                              target_bbox[2], target_bbox[1] + target_bbox[3]), (0, 0, 255), thickness)

                        video_out.write(img)
            toc /= cv2.getTickFrequency()

            if args.vis:
                video_out.release()

            # save results
            #model_path = os.path.join('results', args.dataset, model_name)

            if args.attack_universal:
                model_path = os.path.join('results_Universal_Targeted_{}_{}'.format(
                    attack_method, expcase), args.dataset, str(args.targetcase), model_name)
                #model_path = os.path.join('results_U_{}_{}_{}'.format(attack_method, expcase, args.directions), args.dataset,  str(args.targetcase),model_name)

            else:
                model_path = os.path.join('results_Targeted_{}_{}'.format(attack_method, expcase),
                                          args.dataset, str(args.targetcase), model_name)

            if not os.path.isdir(model_path):
                os.makedirs(model_path)
            result_path = os.path.join(model_path, '{}.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    f.write(','.join([str(i) for i in x]) + '\n')

            target_path = os.path.join(
                model_path, '{}_target.txt'.format(video.name))

            with open(target_path, 'w') as f:
                for x in target_bboxes:
                    f.write(','.join([str(i) for i in x]) + '\n')

            mean_FPS.append(idx / toc)

            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps, Mean Speed: {:3.1f}'.format(
                v_idx + 1, video.name, toc, idx / toc, np.mean(mean_FPS)))

        os.chdir(model_path)
        save_file = '../%s' % dataset
        shutil.make_archive(save_file, 'zip')
        print('Records saved at', save_file + '.zip')


if __name__ == '__main__':
    main()
