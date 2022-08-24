""" Usage

python ttattack_targeted.py  --tracker_name=siamrpn_mobilev2_l234_dwxcorr --dataset=OTB100 --case=2 --gpu=1 --model_iter=4_net_G.pth  --trajcase=SE    --attack_universal --vis
python ttattack_targeted.py  --tracker_name=siamrpn_mobilev2_l234_dwxcorr --dataset=OTB100 --case=2 --gpu=1 --model_iter=4_net_G.pth  --trajcase=SE    --attack_universal #--vis

"""

import argparse
import os
import sys
import cv2
import torch
import time
import math
import datetime
import subprocess
import numpy as np
from tqdm import tqdm
from utils.log import create_logger
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str
from torchvision.utils import save_image

from common_path import *
parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--tracker_name', default=siam_model_, type=str)
parser.add_argument('--dataset', default=dataset_name_, type=str, help='eval one special dataset')
parser.add_argument('--video', default="", type=str, help='eval one special video')
parser.add_argument('--vis', default=False, action='store_true', help='whether visualzie result')
parser.add_argument('--case', type=int, required=True)
parser.add_argument('--gpu', type=str, help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--model_iter', type=str)
parser.add_argument('--eps', type=int, default=0)
parser.add_argument('--istargeted', default=False, action='store_true', help='whether visualzie result')
parser.add_argument('--trajcase', type=str, required=True, help='SE | SW | NE | NW')
parser.add_argument('--targetcase', type=str,)
parser.add_argument('--attack_universal', default=False, action='store_true', help='whether visualzie result')
parser.add_argument('--directions', type=int, default=12)
parser.add_argument('--driftdistance', type=int, default=12)

args = parser.parse_args()
torch.set_num_threads(1)


import os
root_dir = os.path.dirname(os.path.abspath(__file__))


def resizebox(pred_bbox, target_bbox):
    t_cx, t_cy = target_bbox[0] + target_bbox[2] / \
        2, target_bbox[1] + target_bbox[3] / 2
    t_w, t_h = 50, 50
    target_bbox = np.array([t_cx - t_w / 2, t_cy - t_h / 2, t_w, t_h])
    return target_bbox


def rescale_img(im, max_size):

    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    return im


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


def main(cmd_line):

    args.targetcase = args.trajcase

    statsdir = './logs_and_metrics/{}/{}/{}/{}/'.format(args.dataset,
                                                        args.tracker_name, args.case, args.model_iter)
    if not os.path.exists(statsdir):
        os.makedirs(statsdir)

    log_filename = os.path.join(statsdir, 'log_{}.txt'.format(datetime.datetime.now().strftime("%H:%M:%S")))
    log, logclose = create_logger(log_filename=log_filename)
    log("Logger saved at {}".format(log_filename))
    log('Ran experiment with command: "{}"'.format(cmd_line))

    results_dir = './results_Universal_Targeted' if args.attack_universal else 'results_Targeted'

    from GAN_utils_template_1 import get_model_GAN
    GAN, opt = get_model_GAN(log)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    model_name = opt.model + '_{}'.format(args.tracker_name)
    expcase = opt.case
    basedir = './visualizations/'
    model_epoch = opt.model_iter
    opt.TARGETED_ATTACK_RADIUS = args.driftdistance

    log("Eps: {}\nTracker: {}\nModel: {}\nTraj case: {}, Target case: {}, no. of directions: {}, drift distance: {}".format(
        opt.eps, args.tracker_name, args.model_iter, args.trajcase, args.targetcase, args.directions, args.driftdistance))

    st_time = time.time()

    snapshot_path = os.path.join(project_path_, '../tracker_weights//%s/model.pth' % args.tracker_name)
    config_path = os.path.join(project_path_, '../tracker_weights//%s/config.yaml' % args.tracker_name)

    log("Config path: {}".format(config_path))
    log("Snapshot path: {}".format(snapshot_path))

    cfg.merge_from_file(config_path)

    dataset_root = os.path.join(dataset_root_, args.dataset)
    if cfg.TRACK.TYPE in ['DASiamRPNTracker', 'DASiamRPNLTTracker', 'SiamFCTracker']:
        model = None
    else:
        model = ModelBuilder()
        model = load_pretrain(model, snapshot_path).cuda().eval()

    tracker = build_tracker(model, args.dataset)

    dataset = DatasetFactory.create_dataset(name=args.dataset, dataset_root=dataset_root, load_img=False)
    total_lost = 0

    MAE_, SSIM_ = [], []

    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:

        for v_idx, video in enumerate(dataset):

            savedir2 = os.path.join(basedir, args.dataset, str(
                args.case), str(args.trajcase))

            if not os.path.isdir(savedir2):
                os.makedirs(savedir2)

            if args.video != '':
                if video.name != args.video:
                    continue
            frame_counter = 0
            lost_number = 0
            toc = 0
            pred_bboxes = []
            target_bboxes = []
            target_bboxes_rect = []
            pred_bboxes2 = []

            MAE, SSIM, LINF = [], [], []

            max_size = 500
            dir_ = 0

            traj_file = os.path.join(root_dir, "../../../", "targeted_attacks_GT", "{}/{}/{}/".
                                     format(args.dataset, args.targetcase, video.name), video.name + '_001_target.txt')

            with open(traj_file, 'r') as f:
                target_traj = [list(map(float, x.strip().split(','))) for x in f.readlines()]
            target_traj = np.array(target_traj)

            for idx, (img, gt_bbox) in tqdm(enumerate(video)):

                target_bbox = target_traj[idx]
                cx, cy, w, h = get_axis_aligned_bbox(np.array(target_bbox))
                target_bbox_rect = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]

                target_bboxes.append(target_bbox)
                target_bboxes_rect.append(target_bbox_rect)

                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1], gt_bbox[0], gt_bbox[1] + gt_bbox[3] - 1,
                               gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1] + gt_bbox[3] - 1,
                               gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1]]
                tic = cv2.getTickCount()
                if idx == frame_counter:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                    template = tracker.init_adv_T(img, gt_bbox_, GAN)
                    cv2.imwrite(os.path.join(savedir2, video.name + "_template.png"), template)

                    if idx == 0 and args.vis:
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        w, h = img.shape[:2]
                        video_out = cv2.VideoWriter(os.path.join(
                            savedir2, video.name + ".avi"), fourcc, fps=20, frameSize=(h, w))
                        video_search = cv2.VideoWriter(os.path.join(
                            savedir2, video.name + "_search.avi"), fourcc, fps=20, frameSize=(255, 255))

                    pred_bbox = gt_bbox_
                    pred_bboxes.append(1)
                    prev_predbbox = target_bbox_rect
                    pred_bboxes2.append(gt_bbox_)

                elif idx > frame_counter:

                    direction = get_direction(target_bbox_rect, prev_predbbox, idx)[0]
                    outputs = tracker.track_advT(img, GAN, direction)

                    if 0:
                        search_img = outputs['cropx']

                    pred_bbox = outputs['bbox']
                    MAE.append(outputs['metrics']['MAE'].item())
                    SSIM = outputs['metrics']['SSIM']
                    prev_predbbox = pred_bbox
                    pred_bboxes.append(pred_bbox)
                    pred_bboxes2.append(pred_bbox)

                toc += cv2.getTickCount() - tic

                if args.vis and idx > frame_counter:
                    cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 1, 2))], True, (0, 255, 0), 3)

                    bbox = list(map(int, pred_bbox))
                    target_bbox_rect = list(map(int, target_bbox_rect))

                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2],
                                                            bbox[1] + bbox[3]), (0, 255, 255), 3)

                    cv2.rectangle(img, (target_bbox_rect[0], target_bbox_rect[1]), (
                        target_bbox_rect[0] + target_bbox_rect[2], target_bbox_rect[1] + target_bbox_rect[3]), (0, 0, 255), 3)

                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(img, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    video_out.write(img)

            toc /= cv2.getTickFrequency()
            video_path = os.path.join(results_dir, args.dataset, model_name, 'baseline', str(
                expcase), model_epoch, str(args.trajcase), video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(
                video_path, '{}_001.txt'.format(video.name))
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

            MAE_.append(np.mean(MAE))
            log('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}, MAE: {:2.1f}, Avg. Mean: {:2.1f}'
                .format(v_idx + 1, video.name, toc, idx / toc, lost_number, np.mean(MAE), np.mean(MAE_)))
            total_lost += lost_number

            if args.vis:
                video_out.release()

        log("{:s} total lost: {:d}".format(model_name, total_lost))

    else:

        for v_idx, video in enumerate(dataset):
            savedir = os.path.join(basedir, args.dataset, video.name)
            savedir2 = os.path.join(basedir, args.dataset, str(
                args.case), str(args.trajcase))

            if args.vis and not os.path.isdir(savedir2):
                os.makedirs(savedir2)

            if args.video != '':
                if video.name != args.video:
                    continue

            if args.dataset in ['OTB100']:
                traj_file = os.path.join(root_dir, "../../../", "targeted_attacks_GT", "{}/{}/".
                                         format(args.dataset, args.targetcase), video.name + '_target.txt')

            elif args.dataset in ['UAV123']:

                traj_file = os.path.join(root_dir, "../../../", "targeted_attacks_GT", "{}/{}/".
                                         format(args.dataset, args.targetcase), video.name + '_target.txt')

            elif args.dataset in ['lasot']:

                traj_file = os.path.join(root_dir, "../../../", "targeted_attacks_GT", "{}/{}/".
                                         format(args.dataset, args.targetcase), video.name + '_target.txt')

            with open(traj_file, 'r') as f:
                target_traj = [list(map(float, x.strip().split(','))) for x in f.readlines()]

            target_traj = np.array(target_traj)

            max_size = 500
            toc = 0
            pred_bboxes = []
            pred_bboxes2 = []
            target_bboxes = []
            scores = []
            track_times = []
            prev_gt_box_ = None
            dir_ = 0

            for idx, (img, gt_bbox) in tqdm(enumerate(video)):

                target_bbox = target_traj[idx]
                MAE, SSIM, LINF = [], [], []

                tic = cv2.getTickCount()
                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]

                    template = tracker.init_adv_T(img, gt_bbox_, GAN)

                    if args.vis:
                        cv2.imwrite(os.path.join(savedir2, video.name + "_template.png"), template)

                    w, h = img.shape[:2]

                    pred_bbox = gt_bbox_
                    scores.append(None)
                    if 'VOT2018-LT' == args.dataset:
                        pred_bboxes.append([1])
                        pred_bboxes2.append(gt_bbox_)
                        target_bboxes.append(pred_bbox)

                    else:
                        pred_bboxes.append(pred_bbox)
                        target_bboxes.append(pred_bbox)

                    if args.vis:
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        if args.dataset in ['VOT2018-LT', 'UAV123']:
                            img = rescale_img(img, max_size)

                        video_out = cv2.VideoWriter(os.path.join(
                            savedir2, video.name + ".avi"), fourcc, fps=20, frameSize=(img.shape[1], img.shape[0]))

                    prev_predbbox = target_bbox

                else:
                    direction, enhance = get_direction(target_bbox, prev_predbbox, idx)
                    outputs = tracker.track_advT(img, GAN, direction, enhance, frame_index=idx)

                    if 0:
                        search_img = outputs['cropx']
                        perturb_img = outputs['perturb']
                        MAE.append(outputs['metrics']['MAE'].item())
                        SSIM = outputs['metrics']['SSIM']

                    pred_bbox = outputs['bbox']
                    prev_predbbox = pred_bbox
                    pred_bboxes.append(pred_bbox)
                    target_bboxes.append(target_bbox)
                    pred_bboxes2.append(pred_bbox)
                    scores.append(outputs['best_score'])

                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())

                if args.vis and idx > 0:

                    if len(gt_bbox) < 2:
                        continue

                    if math.isnan(gt_bbox[1]):
                        continue

                    gt_bbox = list(map(int, gt_bbox))
                    pred_bbox = list(map(int, pred_bbox))
                    target_bbox = list(map(int, target_bbox))
                    thickness = 2
                    extra_th = 5

                    cv2.rectangle(img, (gt_bbox[0] - extra_th, gt_bbox[1] - extra_th), (gt_bbox[0] +
                                                                                        gt_bbox[2] + extra_th, gt_bbox[1] + gt_bbox[3] + extra_th), (0, 255, 0), thickness)
                    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]), (pred_bbox[0] +
                                                                      pred_bbox[2], pred_bbox[1] + pred_bbox[3]), (0, 255, 255), thickness)
                    cv2.rectangle(img, (target_bbox[0], target_bbox[1]), (target_bbox[0] +
                                                                          target_bbox[2], target_bbox[1] + target_bbox[3]), (0, 0, 255), thickness)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                    if args.dataset in ['VOT2018-LT', 'UAV123']:
                        img = rescale_img(img, max_size)

                    video_out.write(img)

            toc /= cv2.getTickFrequency()

            if args.vis:
                video_out.release()

            # save results
            if 'VOT2018-LT' == args.dataset:
                video_path = os.path.join(results_dir, args.dataset, model_name, 'longterm', str(
                    expcase), model_epoch, str(args.trajcase), video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)

                result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x]) + '\n')

                result_path = os.path.join(video_path, '{}_001_pred.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes2:
                        f.write(','.join([str(i) for i in x]) + '\n')

                result_path = os.path.join(video_path, '{}_001_target.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in target_bboxes:
                        f.write(','.join([str(i) for i in x]) + '\n')

                result_path = os.path.join(video_path, '{}_001_confidence.value'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in scores:
                        f.write('\n') if x is None else f.write(
                            "{:.6f}\n".format(x))
                result_path = os.path.join(video_path, '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))

            elif 'GOT-10k' == args.dataset:
                video_path = os.path.join(results_dir, args.dataset, model_name, video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x]) + '\n')
                result_path = os.path.join(video_path, '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            else:
                model_path = os.path.join(results_dir, args.dataset, model_name,
                                          str(expcase), model_epoch, str(args.trajcase))
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

            MAE_.append(np.mean(MAE))

            log('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps, MAE: {:2.1f}, Avg. Mean: {:2.1f}, Lost:{:4d}'
                .format(v_idx + 1, video.name, toc, idx / toc, np.mean(MAE), np.mean(MAE_), outputs['lost']))

    log("Total time : {:.1f}s, Avg MAE : {:2.1f}".format(time.time() - st_time, np.mean(MAE_)))
    result = subprocess.call(["sh", "-c", " ".join(
        ['python', '-W ignore', 'eval_target.py', '--tracker_path', results_dir, '--dataset', args.dataset,
         '--model_epoch', args.model_iter, '--case', str(args.case), '--tracker_prefix',
         'G_template_L2_500_regress_' + args.tracker_name, '--logfilename', log_filename, '--trajcase', str(args.trajcase), '-ss'])])


if __name__ == '__main__':
    command_line = 'python ' + ' '.join(sys.argv)
    main(command_line)
