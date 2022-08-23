from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import subprocess

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
parser.add_argument('--dataset', type=str, default='UAV123', help='datasets')
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
args = parser.parse_args()

torch.set_num_threads(1)


ckpt_root_dir = '../../SiamRPNpp'


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

        # this is dummy tracker name
        opt.tracker_name = "dimp"
        opt.istargeted = False
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        opt.model = 'G_template_L2_500_regress'
        opt.netG = 'unet_128'
        expcase, model_epoch, opt.eps = args.case, args.model_iter, args.eps

        ckpt = os.path.join(ckpt_root_dir,
                            'checkpoints/{}_{}/{}'.format(opt.model, expcase, model_epoch))
        print("Loading generator trained with TTA approach")

    GAN = create_model(opt)
    print("generator checkpoint path:", ckpt)
    GAN.load_path = ckpt
    GAN.setup(opt)
    GAN.eval()
    return GAN, expcase, attack_method


GAN, expcase, attack_method = load_generator()


def main():
    # load config
    cfg.merge_from_file(args.config)

    # hp_search
    params = getattr(cfg.HP_SEARCH, args.dataset)
    hp = {'lr': params[0], 'penalty_k': params[1], 'window_lr': params[2]}

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(cur_dir, '../testing_dataset', args.dataset)

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
            for idx, (img, gt_bbox) in enumerate(video):
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

                    if idx == 0:
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        w, h = img.shape[:2]
                        video_out = cv2.VideoWriter(os.path.join(
                            "./viz/", video.name + ".avi"), fourcc, fps=20, frameSize=(h, w))

                elif idx > frame_counter:

                    # outputs = tracker.track(img, hp)
                    outputs = tracker.track_advT(img, hp, GAN, 1, frame_id=idx)

                    pred_bbox = outputs['bbox']
                    overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
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

                if args.vis and idx > frame_counter:
                    cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 1, 2))],
                                  True, (0, 255, 0), 3)
                    bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (bbox[0], bbox[1]),
                                  (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(img, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    video_out.write(img)
            toc /= cv2.getTickFrequency()

            if args.vis:
                video_out.release()

            if args.attack_universal:
                video_path = os.path.join('results_U_{}_{}'.format(
                    attack_method, expcase), args.dataset, model_name, 'baseline', video.name)
            else:
                video_path = os.path.join('results_{}_{}'.format(attack_method, expcase),
                                          args.dataset, model_name, 'baseline', video.name)

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
            for idx, (img, gt_bbox) in enumerate(video):
                tic = cv2.getTickCount()
                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                    tracker.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    pred_bboxes.append(pred_bbox)

                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    w, h = img.shape[:2]

                    if args.vis:
                        video_out = cv2.VideoWriter(os.path.join(
                            "./viz/", video.name + ".avi"), fourcc, fps=20, frameSize=(h, w))
                else:

                    # outputs = tracker.track(img, hp)
                    outputs = tracker.track_advT(img, hp, GAN, 1, frame_id=idx)

                    pred_bbox = outputs['bbox']
                    pred_bboxes.append(pred_bbox)
                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())

                # if idx == 0:
                #     cv2.destroyAllWindows()
                if args.vis and idx > 0:
                    if not any(map(math.isnan, gt_bbox)):
                        gt_bbox = list(map(int, gt_bbox))
                        pred_bbox = list(map(int, pred_bbox))
                        cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                      (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]), (0, 255, 0), 3)
                        cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                      (pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3]), (0, 255, 255), 3)
                        cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        video_out.write(img)
            toc /= cv2.getTickFrequency()

            if args.vis:
                video_out.release()

            if args.attack_universal:

                results_dir = 'results_Universal_{}_{}'.format(
                    attack_method, expcase)

                model_path = os.path.join('results_Universal_{}_{}'.format(
                    attack_method, expcase), args.dataset, model_name)
            else:

                results_dir = 'results_TD_{}_{}'.format(
                    attack_method, expcase)

                model_path = os.path.join('results_TD_{}_{}'.format(
                    attack_method, expcase), args.dataset, model_name)

            if not os.path.isdir(model_path):
                os.makedirs(model_path)
            result_path = os.path.join(model_path, '{}.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    f.write(','.join([str(i) for i in x]) + '\n')

            mean_FPS.append(idx / toc)

            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps, Mean Speed: {:3.1f}'.format(
                v_idx + 1, video.name, toc, idx / toc, np.mean(mean_FPS)))

        result = subprocess.call(
            ["sh", "-c", " ".join(
                ['python', '-W ignore', 'eval.py', '--tracker_path', results_dir, '--dataset', args.dataset,
                 '--tracker_prefix', 'model_general'])])

        os.chdir(model_path)
        save_file = '../%s' % dataset
        shutil.make_archive(save_file, 'zip')
        print('Records saved at', save_file + '.zip')


if __name__ == '__main__':
    main()
