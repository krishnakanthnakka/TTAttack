# Copyright (c) SenseTime. All Rights Reserved.

"""

python run_clean.py   --tracker_name=siamrpn_mobilev2_l234_dwxcorr --dataset=lasot --case=22 --gpu=1 --model_iter=8_net_G.pth
python run_clean.py   --tracker_name=siamrpn_mobilev2_l234_dwxcorr --dataset=OTB100 --case=22 --gpu=1 --model_iter=8_net_G.pth

python run_clean.py   --tracker_name=siamrpn_r50_l234_dwxcorr --dataset=UAV123 --case=22 --gpu=1 --model_iter=8_net_G.pth
python run_clean.py   --tracker_name=siamrpn_r50_l234_dwxcorr --dataset=OTB100 --case=22 --gpu=1 --model_iter=8_net_G.pth
python run_clean.py   --tracker_name=siamrpn_r50_l234_dwxcorr --dataset=VOT2018 --case=22 --gpu=1 --model_iter=8_net_G.pth


"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse, subprocess
import os, sys
import math
import cv2
import torch, datetime
import numpy as np
import time
import math
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from utils.log import create_logger

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
parser.add_argument('--trajcase', type=int, default=0)


# parser.add_argument('--gpu', type=str)


args = parser.parse_args()


torch.set_num_threads(1)

#os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu




def get_direction(cur_gt_bbox_, prev_gt_box_, idx):

    x = cur_gt_bbox_[0] - prev_gt_box_[0]
    y = cur_gt_bbox_[1] - prev_gt_box_[1]

    dir_ = (math.atan2(-y, x) + 2 * math.pi) % (2 * math.pi)

    # print(idx, x, y)

    return dir_


def main(cmd_line):

    st_time = time.time()

    snapshot_path = os.path.join(
        project_path_, 'pysot/experiments/%s/model.pth' % args.tracker_name)
    config_path = os.path.join(
        project_path_, 'pysot/experiments/%s/config.yaml' % args.tracker_name)

    statsdir = './logs/{}/{}/{}/{}/'.format(args.dataset,
                                            args.tracker_name, args.case, args.model_iter)
    if not os.path.exists(statsdir):
        os.makedirs(statsdir)


    print("Config path: {}".format(config_path))
    print("snapshot path: {}".format(snapshot_path))
    log_filename=os.path.join(statsdir, 'log_{}.txt'.format(datetime.datetime.now().strftime("%H:%M:%S")))
    log, logclose = create_logger(log_filename)
    log("Logger saved at {}".format(log_filename))
    log('Ran experiment with command: "{}"'.format(cmd_line))


    from GAN_utils_template_1 import get_model_GAN
    GAN, opt = get_model_GAN(log)

    model_name = opt.model
    model_name = opt.model + '_{}'.format(args.tracker_name)
    expcase = opt.case
    model_epoch = opt.model_iter
    basedir = './results_T/'
    results_dir  = 'results_clean'




    # load config
    cfg.merge_from_file(config_path)

    dataset_root = os.path.join(dataset_root_, args.dataset)

    # model = ModelBuilder()
    # model = load_pretrain(model, snapshot_path).cuda().eval()
    # tracker = build_tracker(model)


    if cfg.TRACK.TYPE in ['DASiamRPNTracker', 'DASiamRPNLTTracker', 'SiamFCTracker', 'OceanTracker', 'OceanOnlineTracker']:
        model = None


    # elif cfg.TRACK.TYPE    in ['siamrpn_ban_r50_l234_otb', 'siamrpn_ban_r50_l234' ]:
    #     from pysot.siamban.models.model_builder import ModelBuilder
    #     model = ModelBuilder()
    #     model = load_pretrain(model, snapshot_path).cuda().eval()


    else:

        model = ModelBuilder()
        model = load_pretrain(model, snapshot_path).cuda().eval()

    tracker = build_tracker(model, args.dataset)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    # model_name = args.snapshot.split('/')[-1].split('.')[0]
    total_lost = 0

    MAE_, SSIM_ = [], []

    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
        # restart tracking
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
            frame_counter = 0
            lost_number = 0
            toc = 0
            pred_bboxes = []

            MAE, SSIM, LINF = [], [], []

            dir_ = 0

            for idx, (img, gt_bbox) in enumerate(video):

                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                               gt_bbox[0], gt_bbox[1] + gt_bbox[3] - 1,
                               gt_bbox[0] + gt_bbox[2] -
                               1, gt_bbox[1] + gt_bbox[3] - 1,
                               gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1]]
                tic = cv2.getTickCount()
                if idx == frame_counter:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                    '''GAN'''

                    template = tracker.init_adv_T(img, gt_bbox_, GAN)[0]
                    cv2.imwrite(os.path.join(savedir, str(
                        idx) + "_template.png"), template)

                    if idx == 0 and args.vis:
                        # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        w, h = img.shape[:2]

                        video_out = cv2.VideoWriter(os.path.join(savedir2, video.name + ".avi"),
                                                    fourcc, fps=20, frameSize=(h, w))
                        video_search = cv2.VideoWriter(os.path.join(savedir2, video.name + "_search.avi"),
                                                       fourcc, fps=20, frameSize=(255, 255))

                    # tracker.init_adv(img, gt_bbox_, GAN)

                    pred_bbox = gt_bbox_
                    pred_bboxes.append(1)
                elif idx > frame_counter:

                    outputs = tracker.track(img)

                    #outputs = tracker.track_advT(img, GAN, dir_)

                    search_img = outputs['cropx']
                    pred_bbox = outputs['bbox']
                    #MAE.append(outputs['metrics']['MAE'].item())
                    #SSIM = outputs['metrics']['SSIM']

                    if cfg.MASK.MASK:
                        pred_bbox = outputs['polygon']
                    overlap = vot_overlap(
                        pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                    if overlap > 0:
                        # not lost
                        pred_bboxes.append(pred_bbox)
                    else:
                        # lost object
                        pred_bboxes.append(2)
                        frame_counter = idx + 5  # skip 5 frames
                        lost_number += 1
                else:
                    pred_bboxes.append(0)
                toc += cv2.getTickCount() - tic
                # if idx == 0:
                #     cv2.destroyAllWindows()
                if args.vis and idx > frame_counter:
                    cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 1, 2))],
                                  True, (0, 255, 0), 3)
                    if cfg.MASK.MASK:
                        cv2.polylines(img, [np.array(pred_bbox, np.int).reshape((-1, 1, 2))],
                                      True, (0, 255, 255), 3)
                    else:
                        bbox = list(map(int, pred_bbox))
                        cv2.rectangle(img, (bbox[0], bbox[1]),
                                      (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(img, str(lost_number), (40, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # cv2.imwrite(os.path.join(savedir, str(idx) + ".png"), img)

                    search_img = search_img.data.cpu().numpy(
                    )[0].transpose(1, 2, 0).astype('uint8')
                    # cv2.putText(search_img, str(MAE) + " , " + str(SSIM), (40, 120),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    video_out.write(img)
                    video_search.write(search_img)

                    # search_img = search_img[:, [2, 1, 0], :, :]

                    # save_image(search_img / 255, savedir + "/" + str(idx) + "_search.png")

                    # cv2.imshow(video.name, img)
                    # cv2.waitKey(1)
            toc /= cv2.getTickFrequency()
            # save results

            #    model_path = os.path.join('results', args.dataset, model_name, str(expcase), model_epoch)

            video_path = os.path.join(results_dir, args.dataset, model_name,
                                      'baseline', str(expcase), model_epoch, video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(
                video_path, '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([vot_float2str("%.4f", i)
                                          for i in x]) + '\n')

            MAE_.append(np.mean(MAE))

            # print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps, MAE: {:2.1f}, Avg. Mean: {:2.1f}'.format(
            #     v_idx + 1, video.name, toc, idx / toc, np.mean(MAE), np.mean(MAE_)))
            print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}, MAE: {:2.1f}, Avg. Mean: {:2.1f}'.format(
                v_idx + 1, video.name, toc, idx / toc, lost_number, np.mean(MAE), np.mean(MAE_)))
            total_lost += lost_number

            if args.vis:
                video_out.release()

        print("{:s} total lost: {:d}".format(model_name, total_lost))
    else:
        # OPE tracking
        for v_idx, video in enumerate(dataset):
            savedir = os.path.join(basedir, args.dataset, video.name)
            savedir2 = os.path.join(basedir, args.dataset, str(args.case))

            # print(v_idx, video.name, "calcualting")

            # if v_idx < 210:
            #     print(video.name, "returned")
            #     continue

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

            for idx, (img, gt_bbox) in tqdm(enumerate(video)):

                # if idx == 100:
                #     break

                MAE, SSIM, LINF = [], [], []

                # print(idx)  # , torch.max(img), torch.min(img))
                # if(idx > 10):
                #     exit()
                #print(idx, img.shape)

                tic = cv2.getTickCount()
                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                    '''GAN'''
                    template = tracker.init_adv_T(img, gt_bbox_, GAN)
                    cv2.imwrite(os.path.join(savedir, str(
                        idx) + "_template.png"), template)
                    w, h = img.shape[:2]

                    pred_bbox = gt_bbox_
                    scores.append(None)
                    if 'VOT2018-LT' == args.dataset:
                        pred_bboxes.append([1])
                    else:
                        pred_bboxes.append(pred_bbox)

                    if args.vis:

                        # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                        # fourcc = cv2.VideoWriter_fourcc('d', 'i', 'v', 'x')
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

                        video_out = cv2.VideoWriter(os.path.join(savedir2, video.name + ".avi"),
                                                    fourcc, fps=20, frameSize=(h, w))
                        video_search = cv2.VideoWriter(os.path.join(savedir2, video.name + "_search.avi"),
                                                       fourcc, fps=20, frameSize=(255, 255))

                else:
                    # outputs = tracker.track(img)

                    # cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    # cur_gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]

                    # if(idx > 2):
                    #     dir_ = get_direction(cur_gt_bbox_, prev_gt_box_, idx)

                    dir_ = 1
                    #prev_gt_box_ = cur_gt_bbox_

                    outputs = tracker.track(img)
                    outputs['lost'] = 0
                    print("Lost:", outputs['lost'], end="\r")

                    search_img = outputs['cropx']

                    # MAE.append(outputs['metrics']['MAE'].item())
                    #SSIM = outputs['metrics']['SSIM']

                    MAE.append(0.0)
                    SSIM = 100.0

                    pred_bbox = outputs['bbox']

                    # if cfg.MASK.MASK:
                    #     pred_bbox = outputs['polygon']

                    pred_bboxes.append(pred_bbox)
                    scores.append(outputs['best_score'])

                toc += cv2.getTickCount() - tic
                track_times.append(
                    (cv2.getTickCount() - tic) / cv2.getTickFrequency())
                # if idx == 0:
                #     cv2.destroyAllWindows()

                # video_out.write(img)

                if args.vis and idx > 0:

                    if len(gt_bbox) < 2:
                        continue

                    if math.isnan(gt_bbox[1]):
                        continue

                    gt_bbox = list(map(int, gt_bbox))

                    pred_bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                  (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]), (0, 255, 0), 3)
                    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                  (pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    # cv2.imshow(video.name, img)

                    # cv2.imwrite(os.path.join(savedir, str(idx) + ".png"), img)
                    video_out.write(img)
                    search_img = search_img.data.cpu().numpy(
                    )[0].transpose(1, 2, 0).astype('uint8')
                    # print(search_img.shape)
                    # cv2.putText(search_img, str(MAE) + " , " + str(SSIM), (40, 120),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    video_search.write(search_img)

                    # search_img = search_img[:, [2, 1, 0], :, :]
                    # save_image(search_img / 255, savedir + "/" + str(idx) + "_search.png")

                    # cv2.waitKey(1)

            toc /= cv2.getTickFrequency()

            if args.vis:
                video_out.release()
                video_search.release()

            # save results
            if 'VOT2018-LT' == args.dataset:
                video_path = os.path.join(results_dir, args.dataset, model_name,
                                          'longterm', str(expcase), model_epoch, video.name)
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
                        f.write('\n') if x is None else f.write(
                            "{:.6f}\n".format(x))
                result_path = os.path.join(video_path,
                                           '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            elif 'GOT-10k' == args.dataset:
                video_path = os.path.join(
                    results_dir, args.dataset, model_name, video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(
                    video_path, '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x]) + '\n')
                result_path = os.path.join(video_path,
                                           '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            else:
                model_path = os.path.join(
                    results_dir, args.dataset, model_name, str(expcase), model_epoch)
                # print(model_path)
                if not os.path.isdir(model_path):
                    os.makedirs(model_path)
                result_path = os.path.join(
                    model_path, '{}.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x]) + '\n')

            MAE_.append(np.mean(MAE))

            # print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps, MAE: {:2.1f}, Avg. Mean: {:2.1f}'.format(
            #     v_idx + 1, video.name, toc, idx / toc, np.mean(MAE), np.mean(MAE_)))

            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps, MAE: {:2.1f}, Avg. Mean: {:2.1f}, Lost:{:4d}'.format(
                v_idx + 1, video.name, toc, idx / toc, np.mean(MAE), np.mean(MAE_), outputs['lost']))

    print("Total time : {:.1f}s, Avg MAE : {:2.1f}".format(
        time.time() - st_time, np.mean(MAE_)))

    result = subprocess.call(
            ["sh", "-c", " ".join(
                ['python', '-W ignore', 'eval.py',  '--tracker_path', results_dir, '--dataset', args.dataset,
                 '--model_epoch', args.model_iter,  '--case', str(args.case),  '--tracker_prefix',
                 'G_template_L2_500_regress_'+args.tracker_name,  '--logfilename', log_filename, '-ss'])])




if __name__ == '__main__':
    command_line = 'python ' + ' '.join(sys.argv)
    main(command_line)
