# Copyright (c) SenseTime. All Rights Reserved.

"""


python run_target_direction.py --tracker_name=siamrpn_r50_l234_dwxcorr --dataset=lasot --case=133 --gpu=1 --model_iter=4_net_G.pth --trajcase=24 --offsetx=80 --offsety=80  #--vis


python run_target_direction.py --tracker_name=siamrpn_r50_l234_dwxcorr --dataset=VOT2018 --case=133 --gpu=1 --model_iter=4_net_G.pth --trajcase=30 --offsetx=80 --offsety=80  --vis
python run_target_direction.py --tracker_name=siamrpn_r50_l234_dwxcorr --dataset=OTB100 --case=133 --gpu=1 --model_iter=4_net_G.pth --trajcase=21 --offsetx=80 --offsety=80  --vis
python run_target_direction.py --tracker_name=siamrpn_r50_l234_dwxcorr_lt --dataset=VOT2018-LT --case=133 --gpu=1 --model_iter=4_net_G.pth --trajcase=21 --offsetx=80 --offsety=80  --vis


{21;[3,3], 22:[3, -3]}


"""
DISPX, DISPY = -3, -3
print(DISPX, DISPY)


import argparse
import os
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
parser.add_argument('--gpu', type=str, help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--model_iter', type=str)
parser.add_argument('--eps', type=int, default=0)
parser.add_argument('--istargeted', default=False, action='store_true',
                    help='whether visualzie result')
parser.add_argument('--trajcase', type=int, required=True)
parser.add_argument('--offsetx', type=int, required=True)  # DUMMY
parser.add_argument('--offsety', type=int, required=True)  # DUMMY

# parser.add_argument('--gpu', type=str)
args = parser.parse_args()


torch.set_num_threads(1)
'''GAN'''



def rescale_img(im, max_size):

    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)

    return im


def get_direction(cur_gt_bbox_, prev_gt_box_, idx):

    x = cur_gt_bbox_[0] - prev_gt_box_[0]
    y = cur_gt_bbox_[1] - prev_gt_box_[1]

    dir_ = (math.atan2(-y, x) + 2 * math.pi) % (2 * math.pi)

    # print(idx, x, y)

    return dir_

# initial
# def _bbox_clip(cx, cy, width, height, W, H):
#     cx = max(20, min(cx, W - 20))
#     cy = max(20, min(cy, H - 20))
#     width = max(10, min(width, W))
#     height = max(10, min(height, H))

#     box = [cx, cy, width, height]
#     return box


def _bbox_clip_vot(region, W, H):
    region = np.array(region)
    cx = np.mean(region[0::2])
    cy = np.mean(region[1::2])
    x1 = min(region[0::2])
    x2 = max(region[0::2])
    y1 = min(region[1::2])
    y2 = max(region[1::2])
    A1 = np.linalg.norm(region[0:2] - region[2:4]) * \
        np.linalg.norm(region[2:4] - region[4:6])
    A2 = (x2 - x1) * (y2 - y1)
    s = np.sqrt(A1 / A2)
    w = s * (x2 - x1) + 1
    h = s * (y2 - y1) + 1
    cx = max(20, min(cx, W - 30))
    cy = max(20, min(cy, H - 30))
    width = max(10, min(w, W))
    height = max(10, min(h, H))

    box = [cx - width / 2, cy - height / 2, cx - width / 2, cy + height / 2,
           cx + width / 2, cy + height / 2, cx + width / 2, cy - height / 2, ]
    return box


def _bbox_clip(cx, cy, width, height, W, H):

    cx = cx + width / 2
    cy = cy + height / 2
    cx = max(20, min(cx, W - 20))
    cy = max(20, min(cy, H - 20))
    width = max(10, min(width, W))
    height = max(10, min(height, H))

    box = [cx - width / 2, cy - height / 2, width, height]
    return box


def main():

    statsdir = './logs/{}/{}/{}/{}/'.format(args.dataset, args.tracker_name, args.case, args.model_iter)
    if not os.path.exists(statsdir):
        os.makedirs(statsdir)
    log, logclose = create_logger(log_filename=os.path.join(
        statsdir, 'log_{}.txt'.format(datetime.datetime.now().strftime("%H:%M:%S"))))


    from GAN_utils_template_1 import get_model_GAN
    GAN, opt = get_model_GAN(log)

    model_name = opt.model
    model_name = opt.model + '_{}'.format(args.tracker_name)
    model_epoch = opt.model_iter


    expcase = opt.case

    basedir = './results_T/'

    log("Eps: {}\nTracker:{}\nModel:{}\n".format(opt.eps, args.tracker_name, args.model_iter))





    st_time = time.time()
    snapshot_path = os.path.join(project_path_, 'pysot/experiments/%s/model.pth' % args.tracker_name)
    config_path = os.path.join(project_path_, 'pysot/experiments/%s/config.yaml' % args.tracker_name)
    print("Config path: {}".format(config_path))
    print("snapshot path: {}".format(snapshot_path))
    cfg.merge_from_file(config_path)

    dataset_root = os.path.join(dataset_root_, args.dataset)

    if cfg.TRACK.TYPE in['DASiamRPNTracker', 'DASiamRPNLTTracker', 'SiamFCTracker']:
        model = None
    else:
        model = ModelBuilder()
        model = load_pretrain(model, snapshot_path).cuda().eval()

    tracker = build_tracker(model, args.dataset)

    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    total_lost = 0
    MAE_, SSIM_ = [], []

    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:

        for v_idx, video in tqdm(enumerate(dataset)):

            print(video.name)
            savedir = os.path.join(basedir, args.dataset, video.name)
            savedir2 = os.path.join(basedir, args.dataset, str(args.case), str(args.trajcase))

            if not os.path.isdir(savedir2):
                os.makedirs(savedir2)

            if not os.path.isdir(savedir):
                os.makedirs(savedir)

            if args.video != '':
                if video.name != args.video:
                    continue
            frame_counter = 0
            pred_bboxes = []
            target_bboxes = []
            dir_ = 0

            for idx, (img, gt_bbox) in enumerate(video):

                # print(len(gt_bbox))

                # print(idx)
                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                               gt_bbox[0], gt_bbox[1] + gt_bbox[3] - 1,
                               gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1] + gt_bbox[3] - 1,
                               gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1]]
                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    init_box = gt_bbox.copy()

                    gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    w, h = img.shape[:2]
                    video_out = cv2.VideoWriter(os.path.join(savedir2, video.name + ".avi"),
                                                fourcc, fps=20, frameSize=(h, w))
                    pred_bbox = gt_bbox_
                    pred_bboxes.append(1)
                    print("YO")

                #target_bbox = [x + 80 for x in gt_bbox]

                args.offsetx = (0 * idx)
                args.offsety = (0 * idx)

                # target_bbox = [gt_bbox[0] + args.offsetx, gt_bbox[1] + args.offsety, gt_bbox[2] + args.offsetx,
                #                gt_bbox[3] + args.offsety, gt_bbox[4] + args.offsetx, gt_bbox[5] + args.offsety, gt_bbox[6] + args.offsetx, gt_bbox[7] + args.offsety]

                target_bbox = [init_box[0] + args.offsetx, init_box[1] + args.offsety, init_box[2] + args.offsetx,
                               init_box[3] + args.offsety, init_box[4] + args.offsetx, init_box[5] + args.offsety, init_box[6] + args.offsetx, init_box[7] + args.offsety]

                target_bbox = _bbox_clip_vot(target_bbox, H=img.shape[0], W=img.shape[1])

                target_bboxes.append(target_bbox)

                cx, cy, w, h = get_axis_aligned_bbox(np.array(target_bbox))
                target_bbox_rect = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]

                if args.vis:

                    target_bbox_rect = list(map(int, target_bbox_rect))

                    cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 1, 2))],
                                  True, (0, 255, 0), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.polylines(img, [np.array(target_bbox, np.int).reshape((-1, 1, 2))],
                                  True, (0, 0, 255), 3)

                    cv2.rectangle(img, (target_bbox_rect[0], target_bbox_rect[1]),
                                  (target_bbox_rect[0] + target_bbox_rect[2], target_bbox_rect[1] + target_bbox_rect[3]), (0, 0, 255), 3)

                    video_out.write(img)

            video_path = os.path.join('results', args.dataset, model_name,
                                      'baseline', str(expcase), model_epoch, str(args.trajcase), video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([vot_float2str("%.4f", i) for i in x]) + '\n')

            result_path = os.path.join(video_path, '{}_001_target.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in target_bboxes:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([vot_float2str("%.4f", i) for i in x]) + '\n')
            if args.vis:
                video_out.release()

    else:

        for v_idx, video in enumerate(dataset):
            print(video.name)

            # if(v_idx < 65):
            #     continue

            savedir = os.path.join(basedir, args.dataset, video.name)
            savedir2 = os.path.join(basedir, args.dataset, str(args.case), str(args.trajcase))

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
            target_bboxes = []
            scores = []
            track_times = []
            prev_gt_box_ = None
            dir_ = 0
            max_size = 500

            for idx, (img, gt_bbox) in tqdm(enumerate(video)):

                MAE, SSIM, LINF = [], [], []

                tic = cv2.getTickCount()
                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))

                    init_box = [cx, cy, w, h]

                    gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                    '''GAN'''
                    template = tracker.init_adv_T(img, gt_bbox_, GAN)
                    cv2.imwrite(os.path.join(savedir, str(idx) + "_template.png"), template)
                    w, h = img.shape[:2]

                    pred_bbox = gt_bbox_
                    scores.append(None)
                    if 'VOT2018-LT' == args.dataset:
                        pred_bboxes.append([1])
                        target_bboxes.append(pred_bbox)

                    else:
                        pred_bboxes.append(pred_bbox)
                        target_bboxes.append(pred_bbox)

                    if args.vis:

                        fourcc = cv2.VideoWriter_fourcc(*'XVID')

                        if args.dataset in ['VOT2018-LT', 'UAV123']:
                            # print(img.shape)
                            img = rescale_img(img, max_size)
                        # print(img.shape)
                        # exit()

                        video_out = cv2.VideoWriter(os.path.join(savedir2, video.name + ".avi"),
                                                    fourcc, fps=20, frameSize=(img.shape[1], img.shape[0]))

                        video_search = cv2.VideoWriter(os.path.join(savedir2, video.name + "_search.avi"),

                                                       fourcc, fps=20, frameSize=(255, 255))

                else:
                    # outputs = tracker.track(img)

                    if(len(gt_bbox) < 4):
                        gt_bbox = prev_gt_box_
                        # prev_gt_box_
                        target_bboxes.append(prev_gt_box_)

                    else:

                        cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                        cur_gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]

                        # if(idx > 2):
                        #     dir_ = get_direction(cur_gt_bbox_, prev_gt_box_, idx)

                        # CHANGED THIS INSTEAD OF w,h
                        #cx, cy, w, h = [cx - (w - 1) / 2 + args.offsetx, cy - (h - 1) / 2 + args.offsety, w, h]

                        cx, cy, w, h = [init_box[0] + (DISPX * idx), init_box[1] + (DISPY * idx), init_box[2], init_box[3]]

                        #print(cx, cy, w, h, idx)

                        cx, cy, w, h = _bbox_clip(cx, cy, w, h, H=img.shape[0], W=img.shape[1])

                        # prev_gt_box_ = [cx - (w - 1) / 2 +  80, cy - (h - 1) / 2 + 80, w, h]

                        prev_gt_box_ = [cx, cy, w, h]

                        # print(img.shape)

                        # exit()

                        target_bbox = prev_gt_box_

                        target_bboxes.append(target_bbox)

                    # scores.append(outputs['best_score'])

                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())

                if args.vis and idx > 0:

                    if len(gt_bbox) < 2:
                        continue

                    if math.isnan(gt_bbox[1]):
                        continue

                    gt_bbox = list(map(int, gt_bbox))
                    target_bbox = list(map(int, target_bbox))

                    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                  (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]), (0, 255, 0), 3)

                    cv2.rectangle(img, (target_bbox[0], target_bbox[1]),
                                  (target_bbox[0] + target_bbox[2], target_bbox[1] + target_bbox[3]), (0, 0, 255), 3)

                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                    if args.dataset in ['VOT2018-LT', 'UAV123']:
                        img = rescale_img(img, max_size)

                    video_out.write(img)

            toc /= cv2.getTickFrequency()

            if args.vis:
                video_out.release()
                video_search.release()

            # save results
            if 'VOT2018-LT' == args.dataset:
                video_path = os.path.join('results', args.dataset, model_name,
                                          'longterm', str(expcase), model_epoch, str(args.trajcase), video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)

                result_path = os.path.join(video_path,
                                           '{}_001_target.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in target_bboxes:
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
                video_path = os.path.join('results', args.dataset, model_name, video.name)
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
                model_path = os.path.join('results_target', args.dataset, model_name,
                                          str(expcase), model_epoch, str(args.trajcase))
                # print(model_path)
                if not os.path.isdir(model_path):
                    os.makedirs(model_path)
                result_path = os.path.join(model_path, '{}.txt'.format(video.name))

                target_path = os.path.join(model_path, '{}_target.txt'.format(video.name))
                with open(target_path, 'w') as f:
                    for x in target_bboxes:
                        f.write(','.join([str(i) for i in x]) + '\n')


if __name__ == '__main__':
    main()
