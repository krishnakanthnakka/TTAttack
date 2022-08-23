""" Usage

python tt_attack_targeted.py  --tracker_name=siam_ocean_online --dataset=OTB100 --case=2 --gpu=1 --model_iter=4_net_G.pth  --trajcase=11  --targetcase=11  --attack_universal


"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import sys
import math
import cv2
import torch
import numpy as np
import time
import math
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from utils.log import create_logger
import datetime
import subprocess


from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker_online
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str
from torchvision.utils import save_image

from common_path import *
parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--tracker_name', default=siam_model_, type=str)
parser.add_argument('--dataset', default=dataset_name_, type=str,
                    help='eval one special dataset')
parser.add_argument('--video', default="", type=str,
                    help='eval one special video')
parser.add_argument('--vis', default=False, action='store_true',
                    help='whether visualzie result')
parser.add_argument('--case', type=int, required=True)
parser.add_argument('--gpu', type=str,
                    help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--model_iter', type=str)
parser.add_argument('--eps', type=int)
parser.add_argument('--istargeted', default=False, action='store_true', help='whether visualzie result')
parser.add_argument('--trajcase', type=int, default=0)
parser.add_argument('--attack_universal', default=False, action='store_true',
                    help='whether visualzie result')
parser.add_argument('--targetcase', type=int, required=True)

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


def main(cmd_line):

    statsdir = './logs_and_metrics/{}/{}/{}/{}/'.format(args.dataset,
                                                        args.tracker_name, args.case, args.model_iter)
    if not os.path.exists(statsdir):
        os.makedirs(statsdir)

    log_filename = os.path.join(statsdir, 'log_{}.txt'.format(datetime.datetime.now().strftime("%H:%M:%S")))
    log, logclose = create_logger(log_filename)
    log("Logger saved at {}".format(log_filename))
    log('Ran experiment with command: "{}"'.format(cmd_line))

    from GAN_utils_template_1 import get_model_GAN
    GAN, opt = get_model_GAN(log)

    model_name = opt.model
    model_name = opt.model + '_{}'.format(args.tracker_name)
    expcase = opt.case
    model_epoch = opt.model_iter

    results_dir = './results_Universal_Targeted' if args.attack_universal else 'results_TD_Targeted'

    basedir = './visualizations/'

    log("Eps: {}\nTracker:{}\nModel:{}\n".format(
        opt.eps, args.tracker_name, args.model_iter))

    st_time = time.time()

    snapshot_path = os.path.join(
        project_path_, '../tracker_weights//%s/model.pth' % args.tracker_name)
    config_path = os.path.join(
        project_path_, '../tracker_weights//%s/config.yaml' % args.tracker_name)

    print("Config path from eval file: {}".format(config_path))
    print("snapshot path from eval file: {}".format(snapshot_path))
    print("Dataset:{}".format(args.dataset))

    cfg.merge_from_file(config_path)
    dataset_root = os.path.join(dataset_root_, args.dataset)

    if cfg.TRACK.TYPE in ['DASiamRPNTracker', 'DASiamRPNLTTracker', 'SiamFCTracker', 'OceanTracker', 'OceanOnlineTracker']:
        model = None

    else:

        model = ModelBuilder()
        model = load_pretrain(model, snapshot_path).cuda().eval()

    ckpt_dict = {'OTB100': 'OceanOon.model', 'VOT2018': 'OceanV18on.pth',
                 'UAV123': 'OceanOon.model', 'lasot': 'OceanOon.model'}
    tracker, online_tracker = build_tracker_online(
        model, args.dataset, checkpoint_name=ckpt_dict[args.dataset], online=True)

    dataset = DatasetFactory.create_dataset(name=args.dataset, dataset_root=dataset_root, load_img=False)
    total_lost = 0
    MAE_, SSIM_ = [], []

    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:

        for v_idx, video in enumerate(dataset):
            savedir = os.path.join(basedir, args.dataset, video.name)
            savedir2 = os.path.join(basedir, args.dataset, str(args.case))

            if not os.path.isdir(savedir2):
                os.makedirs(savedir2)

            if not os.path.isdir(savedir):
                os.makedirs(savedir)

            if args.video != '':
                if video.name != args.video:
                    continue
            frame_counter = 0
            lost_number = 0
            toc = 0
            pred_bboxes = []

            MAE, SSIM, LINF = [], [], []

            dir_ = 0
            target_bboxes = []
            target_bboxes_rect = []
            pred_bboxes2 = []

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
                    gt_bbox = [gt_bbox[0], gt_bbox[1], gt_bbox[0], gt_bbox[1] + gt_bbox[3] - 1,
                               gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1] + gt_bbox[3] - 1,
                               gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1]]
                tic = cv2.getTickCount()

                if idx == frame_counter:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                    recompute = False if idx != 0 else True

                    #template = tracker.init_adv_T(img, gt_bbox_, GAN, recompute)

                    template, state = tracker.init_adv_T(img, gt_bbox_, GAN)
                    online_tracker.init_adv_T(img, tracker.siam_net, gt_bbox_, True,
                                              dataname=args.dataset, resume='snapshot/{}'.format(ckpt_dict[args.dataset]))

                    cv2.imwrite(os.path.join(savedir, str(idx) + "_template.png"), template)

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
                    outputs = online_tracker.track_advT(img, state, GAN, tracker, idx, direction=direction)

                    state = outputs['state']
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
                    if cfg.MASK.MASK:
                        cv2.polylines(img, [np.array(pred_bbox, np.int).reshape(
                            (-1, 1, 2))], True, (0, 255, 255), 3)
                    else:
                        bbox = list(map(int, pred_bbox))
                        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2],
                                                                bbox[1] + bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(img, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    search_img = search_img.data.cpu().numpy(
                    )[0].transpose(1, 2, 0).astype('uint8')

                    video_out.write(img)
                    video_search.write(search_img)
            toc /= cv2.getTickFrequency()
            # save results

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
        # OPE tracking
        for v_idx, video in enumerate(dataset):
            savedir = os.path.join(basedir, args.dataset, video.name)
            savedir2 = os.path.join(basedir, args.dataset, str(args.case))

            if not os.path.isdir(savedir2):
                os.makedirs(savedir2)

            if not os.path.isdir(savedir):
                os.makedirs(savedir)

            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            toc = 0
            pred_bboxes = []
            scores = []
            track_times = []
            prev_gt_box_ = None
            dir_ = 0
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

            # elif args.dataset in ['lasot']:
            #     traj_file = os.path.join("/cvlabdata1/home/krishna/AttTracker/pysot/tools/results_target/{}/G_template_L2_500_regress_siamrpn_r50_l234_dwxcorr/133/4_net_G.pth/{}/".
            #                              format(args.dataset, args.targetcase), video.name + '_target.txt')

            if args.dataset in ['OTB100']:
                traj_file = os.path.join(root_dir, "../", "targeted_attacks_GT", "{}/{}/".
                                         format(args.dataset, args.targetcase), video.name + '_target.txt')

            elif args.dataset in ['UAV123']:

                traj_file = os.path.join(root_dir, "../../../", "targeted_attacks_GT", "{}/{}/".
                                         format(args.dataset, args.targetcase), video.name + '_target.txt')

                # traj_file = os.path.join("/cvlabdata1/home/krishna/AttTracker/pysot/tools/results_paper/{}/G_template_L2_500_regress_siamrpn_r50_l234_dwxcorr2/133/4_net_G.pth/{}/".
                #                          format(args.dataset, args.targetcase), video.name + '_target.txt')

            elif args.dataset in ['lasot']:

                traj_file = os.path.join(root_dir, "../../../", "targeted_attacks_GT", "{}/{}/".
                                         format(args.dataset, args.targetcase), video.name + '_target.txt')

            with open(traj_file, 'r') as f:
                target_traj = [list(map(float, x.strip().split(','))) for x in f.readlines()]

            target_traj = np.array(target_traj)

            for idx, (img, gt_bbox) in tqdm(enumerate(video)):

                target_bbox = target_traj[idx]

                MAE, SSIM, LINF = [], [], []
                rgb_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                tic = cv2.getTickCount()

                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]

                    template, state = tracker.init_adv_T(img, gt_bbox_, GAN)
                    online_tracker.init_adv_T(img, tracker.siam_net, gt_bbox_, load_online=True, dataname=args.dataset,
                                              resume='{}'.format(ckpt_dict[args.dataset]))

                    w, h = img.shape[:2]
                    pred_bbox = gt_bbox_
                    scores.append(None)
                    if 'VOT2018-LT' == args.dataset:
                        pred_bboxes.append([1])
                    else:
                        pred_bboxes.append(pred_bbox)
                        target_bbox = target_traj[idx]
                        target_bboxes.append(target_bbox)

                    if args.vis:
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        video_out = cv2.VideoWriter(os.path.join(
                            savedir2, video.name + ".avi"), fourcc, fps=20, frameSize=(h, w))
                        video_search = cv2.VideoWriter(os.path.join(
                            savedir2, video.name + "_search.avi"), fourcc, fps=20, frameSize=(255, 255))
                    target_bbox = target_traj[idx]
                    prev_predbbox = target_bbox

                else:

                    #outputs = online_tracker.track(img,  siam_tracker, state)
                    direction, enhance = get_direction(target_bbox, prev_predbbox, idx)
                    outputs = online_tracker.track_advT(img, state, GAN, tracker, idx, direction=direction)

                    state = outputs['state']
                    search_img = outputs['cropx']
                    MAE.append(outputs['metrics']['MAE'].item())
                    SSIM = outputs['metrics']['SSIM']
                    pred_bbox = outputs['bbox']
                    target_bboxes.append(target_bbox)

                    pred_bboxes.append(pred_bbox)
                    scores.append(outputs['best_score'])
                    prev_predbbox = pred_bbox

                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())

                if args.vis and idx > 0:

                    if len(gt_bbox) < 2:
                        continue
                    if math.isnan(gt_bbox[1]):
                        continue
                    target_bbox = list(map(int, target_bbox))
                    gt_bbox = list(map(int, gt_bbox))
                    pred_bbox = list(map(int, pred_bbox))

                    thickness = 2
                    extra_th = 5

                    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]), (gt_bbox[0] +
                                                                  gt_bbox[2], gt_bbox[1] + gt_bbox[3]), (0, 255, 0), 3)
                    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]), (pred_bbox[0] +
                                                                      pred_bbox[2], pred_bbox[1] + pred_bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.rectangle(img, (target_bbox[0], target_bbox[1]), (target_bbox[0] +
                                                                          target_bbox[2], target_bbox[1] + target_bbox[3]), (0, 0, 255), thickness)

                    # cv2.imwrite(os.path.join(savedir, str(idx) + ".png"), img)
                    video_out.write(img)
                    search_img = search_img.data.cpu().numpy()[0].transpose(1, 2, 0).astype('uint8')
                    # cv2.putText(search_img, str(MAE) + " , " + str(SSIM), (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    video_search.write(search_img)
                    # search_img = search_img[:, [2, 1, 0], :, :]
                    # save_image(search_img / 255, savedir + "/" + str(idx) + "_search.png")

            toc /= cv2.getTickFrequency()

            if args.vis:
                video_out.release()
                video_search.release()

            model_path = os.path.join(results_dir, args.dataset, model_name,
                                      str(expcase), model_epoch, str(args.trajcase))
            # print(model_path)
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

            log('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps, MAE: {:2.1f}, Avg. Mean: {:2.1f}, Lost:{:4d}'.format(
                v_idx + 1, video.name, toc, idx / toc, np.mean(MAE), np.mean(MAE_), outputs['lost']))

    log("Total time : {:.1f}s, Avg MAE : {:2.1f}".format(
        time.time() - st_time, np.mean(MAE_)))

    result = subprocess.call(
        ["sh", "-c", " ".join(
            ['python', '-W ignore', 'eval_target.py', '--tracker_path', results_dir, '--dataset', args.dataset,
             '--model_epoch', args.model_iter, '--case', str(args.case), '--tracker_prefix',
             'G_template_L2_500_regress_' + args.tracker_name, '--logfilename', log_filename, '--trajcase', '11', '-ss'])])


if __name__ == '__main__':

    command_line = 'python ' + ' '.join(sys.argv)
    main(command_line)
