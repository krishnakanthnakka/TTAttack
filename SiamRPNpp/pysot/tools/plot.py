"""
python plot.py --tracker_path=./paper_plots --dataset=UAV123 --model_epoch=1_net_G.pth --case=1 --trajcase=1 --model_epoch=1_net_G.pth --vis
python plot.py --tracker_path=./paper_plots --dataset=VOT2018 --model_epoch=1_net_G.pth --case=4 --trajcase=1 --model_epoch=4_net_G.pth --vis

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
from toolkit.visualization.draw_success_precision import draw_success_precision
import pickle

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
from toolkit.datasets import OTBDataset, UAVDataset, LaSOTDataset, \
    VOTDataset, NFSDataset, VOTLTDataset
from toolkit.evaluation import OPEBenchmark, AccuracyRobustnessBenchmark, \
    EAOBenchmark, F1Benchmark
from utils.log import create_logger
from toolkit.visualization.draw_success_precision import draw_success_precision
# from toolkit.visualization.draw_eao import draw_eao
# from toolkit.visualization.draw_f1 import draw_f1


import sys

import datetime

parser = argparse.ArgumentParser(description='tracking evaluation')
parser.add_argument('--tracker_path', '-p', type=str,
                    help='tracker result path')
parser.add_argument('--dataset', '-d', type=str,
                    help='dataset name')
parser.add_argument('--num', '-n', default=1, type=int,
                    help='number of thread to eval')
parser.add_argument('--tracker_prefix', '-t', default='',
                    type=str, help='tracker name')
parser.add_argument('--show_video_level', '-s', dest='show_video_level',
                    action='store_true')

parser.add_argument('--model_epoch', '-m', type=str, help='depochmodel')
parser.add_argument('--case', '-c', type=int, help='depochmodel')
parser.add_argument('--vis', dest='vis', action='store_true')
parser.add_argument('--trajcase', type=int, required=True)

parser.set_defaults(show_video_level=False)
args = parser.parse_args()


def main():

    tracker_dir = os.path.join(args.tracker_path, args.dataset)
    trackers = glob(os.path.join(args.tracker_path, args.dataset, '*'))
    trackers = [x.split('/')[-1] for x in trackers]
    root = os.path.realpath(os.path.join(os.path.dirname(__file__),
                                         '../testing_dataset'))
    root = os.path.join(root, args.dataset)

    dataset = UAVDataset(args.dataset, root)
    dataset.set_tracker(tracker_dir, trackers)

    dataset = UAVDataset(args.dataset, root)
    dataset.set_tracker(tracker_dir, trackers)

    benchmark = OPEBenchmark(dataset, model_epoch=args.model_epoch,
                             expcase=args.case)

    success_ret = {}
    with Pool(processes=args.num) as pool:
        for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                                            trackers), desc='eval success', total=len(trackers), ncols=100):
            success_ret.update(ret)
    precision_ret = {}
    with Pool(processes=args.num) as pool:
        for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                                            trackers), desc='eval precision', total=len(trackers), ncols=100):
            precision_ret.update(ret)

    norm_precision_ret = {}

    with Pool(processes=args.num) as pool:
        for ret in tqdm(pool.imap_unordered(benchmark.eval_norm_precision,
                                            trackers), desc='eval precision', total=len(trackers), ncols=100):
            norm_precision_ret.update(ret)

    # print(success_ret)
    # exit()

    if args.vis:
        for attr, videos in dataset.attr.items():
            # print(attr)
            if attr == 'ALL':
                draw_success_precision(success_ret,
                                       name=dataset.name,
                                       videos=videos,
                                       attr=attr,
                                       precision_ret=precision_ret, norm_precision_ret=norm_precision_ret)

    #draw_success_precision(success_ret,name=dataset.name,  videos=videos,  attr='ALL',  precision_ret=precision_ret)


if __name__ == '__main__':
    main()
