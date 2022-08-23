# Copyright (c) SenseTime. All Rights Reserved.

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
import subprocess

from siamban.core.config import cfg
from siamban.models.model_builder import ModelBuilder
from siamban.tracker.tracker_builder import build_tracker
from siamban.utils.bbox import get_axis_aligned_bbox
from siamban.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str


parser = argparse.ArgumentParser(description='siamese tracking')
parser.add_argument('--dataset', type=str,
                    help='datasets')
parser.add_argument('--config', default='', type=str,
                    help='config file')
parser.add_argument('--snapshot', default='', type=str,
                    help='snapshot of models to eval')
parser.add_argument('--video', default='', type=str,
                    help='eval one special video')
parser.add_argument('--vis', action='store_true',
                    help='whether visualzie result')
parser.add_argument('--gpu_id', default='not_set', type=str,
                    help="gpu id")
parser.add_argument('--case', type=int, required=True)
parser.add_argument('--model_iter', type=str, required=True)
parser.add_argument('--eps', type=int, required=True)
parser.add_argument('--attack_universal', default=False, action='store_true',
                    help='whether visualzie result')

parser.add_argument('--trajcase', type=int, required=True)
parser.add_argument('--targetcase', type=int, required=True)
parser.add_argument('--istargeted', default=False, action='store_true', help='whether visualzie result')
parser.add_argument('--directions', type=int, default=12)

args = parser.parse_args()

if args.gpu_id != 'not_set':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

torch.set_num_threads(1)


import os
root_dir = os.path.dirname(os.path.abspath(__file__))


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


ckpt_root_dir = '../../../SiamRPNpp'


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
        # opt.tracker_name = "siamrpn_r50_l234_dwxcorr"  # HACK. CHANGED FROM dimp
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
    cfg.merge_from_file(args.config)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(cur_dir, '../../testing_dataset', args.dataset)

    # create model
    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    tracker = build_tracker(model)
    basedir = './viz/'

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    model_name = args.snapshot.split('/')[-1].split('.')[0]
    total_lost = 0
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
            pred_bboxes2 = []
            target_bboxes_rect = []
            target_bboxes = []

            if args.case == 133 and args.trajcase in [11, 12, 13, 14, 21, 22, 23, 24]:
                assert False, "DONT RUN THIS"

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
                    pred_bbox = gt_bbox_
                    prev_predbbox = target_bbox_rect
                    pred_bboxes2.append(gt_bbox_)

                    if idx == 0 and args.vis:
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        w, h = img.shape[:2]
                        video_out = cv2.VideoWriter(os.path.join(
                            "/cvlabdata1/home/krishna/AttTracker/baselines/siamban/viz/", video.name + ".avi"), fourcc, fps=20, frameSize=(h, w))
                        #video_search = cv2.VideoWriter(os.path.join(savedir2, video.name + "_search.avi"), fourcc, fps=20, frameSize=(255, 255))

                elif idx > frame_counter:

                    #outputs = tracker.track(img)
                    direction = get_direction(target_bbox_rect, prev_predbbox, idx)[0]
                    outputs = tracker.track_advT(img, GAN, direction, frame_id=idx)

                    pred_bbox = outputs['bbox']
                    prev_predbbox = pred_bbox

                    pred_bboxes.append(pred_bbox)
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
            # save results

            # video_path = os.path.join('results_{}_{}'.format(attack_method, expcase), args.dataset, model_name,
            #         'baseline', video.name)

            if args.attack_universal:
                video_path = os.path.join('results_U_{}_{}'.format(attack_method, expcase), args.dataset, str(
                    args.targetcase), model_name, 'baseline', video.name)
            else:
                video_path = os.path.join('results_{}_{}'.format(attack_method, expcase), args.dataset, str(
                    args.targetcase), model_name, 'baseline', video.name)

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

            savedir2 = os.path.join(basedir, args.dataset, str(
                args.case), str(args.trajcase))

            # if v_idx< 160:
            #     continue

            if args.vis and not os.path.isdir(savedir2):
                os.makedirs(savedir2)

            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            toc = 0
            pred_bboxes = []
            scores = []
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
                    scores.append(None)
                    if 'VOT2018-LT' == args.dataset:
                        pred_bboxes.append([1])
                    else:
                        pred_bboxes.append(pred_bbox)
                        target_bboxes.append(pred_bbox)

                    # print(args.vis)

                    if args.vis:
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        w, h = img.shape[:2]
                        video_out = cv2.VideoWriter(os.path.join(
                            savedir2, video.name + ".avi"), fourcc, fps=20, frameSize=(h, w))
                        #video_search = cv2.VideoWriter(os.path.join("./viz/", video.name + "_search.avi"), fourcc, fps=20, frameSize=(255, 255))
                        #video_perturb = cv2.VideoWriter(os.path.join("./viz/", video.name + "_perturb.avi"), fourcc, fps=20, frameSize=(255, 255))

                    prev_predbbox = target_bbox

                else:
                    #outputs = tracker.track(img)
                    direction, enhance = get_direction(target_bbox, prev_predbbox, idx)
                    outputs = tracker.track_advT(img, GAN, direction, frame_id=idx)
                    target_bboxes.append(target_bbox)

                    pred_bbox = outputs['bbox']
                    pred_bboxes.append(pred_bbox)
                    scores.append(outputs['best_score'])
                    prev_predbbox = pred_bbox

                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())
                # if idx == 0:
                #     cv2.destroyAllWindows()
                if args.vis and idx > 0:
                    gt_bbox = list(map(int, gt_bbox))
                    pred_bbox = list(map(int, pred_bbox))
                    target_bbox = list(map(int, target_bbox))
                    thickness = 2
                    extra_th = 5

                    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                  (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]), (0, 255, 0), 3)
                    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                  (pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                    cv2.rectangle(img, (target_bbox[0], target_bbox[1]), (target_bbox[0] +
                                                                          target_bbox[2], target_bbox[1] + target_bbox[3]), (0, 0, 255), thickness)

                    #cv2.imshow(video.name, img)
                    # cv2.waitKey(1)
                    video_out.write(img)

            if args.vis:
                video_out.release()
                # video_search.release()

            toc /= cv2.getTickFrequency()
            # save results
            if 'VOT2018-LT' == args.dataset:
                video_path = os.path.join('results_{}_{}'.format(attack_method, expcase), args.dataset, model_name,
                                          'longterm', video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path,
                                           '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x]) + '\n')
                result_path = os.path.join(video_path,
                                           '{}_001_confidence.value'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in scores:
                        f.write('\n') if x is None else f.write("{:.6f}\n".format(x))
                result_path = os.path.join(video_path,
                                           '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            elif 'GOT-10k' == args.dataset:
                #video_path = os.path.join('results_{}_{}'.format(attack_method, expcase), args.dataset, model_name, video.name)
                #video_path = os.path.join('results_{}_{}'.format(attack_method, expcase), args.dataset, model_name)

                if args.attack_universal:
                    model_path = os.path.join('results_U_{}_{}'.format(
                        attack_method, expcase), args.dataset, model_name)
                else:
                    model_path = os.path.join('results_{}_{}'.format(
                        attack_method, expcase), args.dataset, model_name)

                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x]) + '\n')
                result_path = os.path.join(video_path,
                                           '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            else:
                #model_path = os.path.join('results_{}_{}'.format(attack_method, expcase), args.dataset, model_name)

                if args.attack_universal:

                    results_dir = 'results_Universal_Targeted_{}_{}'.format(
                        attack_method, expcase)
                    #model_path = os.path.join('results_U_{}_{}'.format(attack_method, expcase), args.dataset, str(args.targetcase), model_name)
                    model_path = os.path.join('results_Universal_Targeted_{}_{}'.format(
                        attack_method, expcase), args.dataset, str(args.targetcase), model_name)  # CHANGED

                else:

                    results_dir = 'results_Targeted_{}_{}'.format(
                        attack_method, expcase, args.directions)
                    model_path = os.path.join('results_Targeted_{}_{}'.format(
                        attack_method, expcase), args.dataset, str(args.targetcase), model_name)

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

            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx + 1, video.name, toc, idx / toc))

    result = subprocess.call(["sh", "-c", " ".join(
        ['python', '-W ignore', '../../tools/eval_target.py', '--tracker_path', results_dir, '--dataset', args.dataset,
         '--num', str(1), '--tracker_prefix', 'model', '--trajcase', str(args.trajcase)])])


if __name__ == '__main__':
    main()
