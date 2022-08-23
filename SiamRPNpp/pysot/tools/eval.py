""" For evluating the results on Untargeted Attacks.

python eval.py --tracker_path=./results_universal --dataset=OTB100 --model_epoch=4_net_G.pth --case=1  --tracker_prefix=G_template_L2_500_regress_siamrpn_mobilev2_l234_dwxcorr

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import datetime
import matplotlib.pyplot as plt
import os
import pickle
import sys

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
from utils.log import create_logger
from toolkit.visualization.draw_success_precision import draw_success_precision
from toolkit.datasets import (OTBDataset, UAVDataset, LaSOTDataset,
                              VOTDataset, NFSDataset, VOTLTDataset)
from toolkit.evaluation import (OPEBenchmark, AccuracyRobustnessBenchmark,
                                EAOBenchmark, F1Benchmark)


parser = argparse.ArgumentParser(description='tracking evaluation')
parser.add_argument('--tracker_path', '-p', type=str, help='tracker result path')
parser.add_argument('--dataset', '-d', type=str, help='dataset name')
parser.add_argument('--num', '-n', default=1, type=int, help='number of thread to eval')
parser.add_argument('--tracker_prefix', '-t', default='', type=str, help='tracker name')
parser.add_argument('--show_video_level', '-s', dest='show_video_level', action='store_true')
parser.add_argument('--model_epoch', '-m', type=str, help='depochmodel')
parser.add_argument('--case', '-c', type=int, help='depochmodel')
parser.add_argument('--vis', dest='vis', action='store_true')
parser.set_defaults(show_video_level=False)
parser.add_argument('--logfilename', dest='logfilename', type=str, default='')

args = parser.parse_args()


def main():

    tracker_dir = os.path.join(args.tracker_path, args.dataset)
    if args.tracker_prefix != '':
        trackers = os.path.join(args.tracker_path, args.dataset, args.tracker_prefix)
        trackers = [args.tracker_prefix]
    else:
        trackers = glob(os.path.join(args.tracker_path, args.dataset, args.tracker_prefix + '*'))
        trackers = [x.split('/')[-1] for x in trackers]

    assert len(trackers) > 0

    args.num = min(args.num, len(trackers))
    root = os.path.realpath(os.path.join(os.path.dirname(__file__), '../../../testing_dataset'))
    root = os.path.join(root, args.dataset)

    if args.logfilename == '':
        log, logclose = create_logger(log_filename=os.path.join('./logs_and_metrics/{}/'.format(args.dataset), 'log_{}.txt'
                                                                .format(datetime.datetime.now().strftime("%H:%M:%S"))))

    else:
        log, logclose = create_logger(log_filename=args.logfilename, log_append=True)

    log("Trackers: {}".format(trackers))
    log("Case: {}".format(args.case))
    log("Checkpoint: {}".format(args.model_epoch))
    log("Dataset: {}".format(args.dataset))

    if args.dataset in ['OTB100', 'OTB2', 'GOT10KVal', 'lasot']:

        dataset = OTBDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset, model_epoch=args.model_epoch, expcase=args.case)
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

        benchmark.show_result(success_ret, precision_ret, norm_precision_ret,
                              show_video_level=args.show_video_level, log=log)
        if args.vis:
            for attr, videos in dataset.attr.items():
                if attr == 'ALL':
                    draw_success_precision(success_ret, name=dataset.name, videos=videos,
                                           attr=attr, precision_ret=precision_ret)

    elif 'LaSOT' == args.dataset:
        dataset = LaSOTDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
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
                                                trackers), desc='eval norm precision', total=len(trackers), ncols=100):
                norm_precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret, norm_precision_ret,
                              show_video_level=args.show_video_level, log=log)

    elif 'UAV' in args.dataset:
        dataset = UAVDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset, model_epoch=args.model_epoch, expcase=args.case)
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
        benchmark.show_result(success_ret, precision_ret,
                              show_video_level=args.show_video_level, log=log)

    elif 'NFS' in args.dataset:
        dataset = NFSDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                                                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision, trackers), desc='eval precision', total=len(trackers),
                            ncols=100):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                              show_video_level=args.show_video_level)
    elif args.dataset in ['VOT2016', 'VOT2017', 'VOT2018', 'VOT2019']:
        dataset = VOTDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        ar_benchmark = AccuracyRobustnessBenchmark(
            dataset, model_epoch=args.model_epoch, expcase=args.case)
        ar_result = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(ar_benchmark.eval, trackers), desc='eval ar', total=len(trackers), ncols=100):
                ar_result.update(ret)

        benchmark = EAOBenchmark(dataset, model_epoch=args.model_epoch, expcase=args.case)
        eao_result = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval, trackers), desc='eval eao', total=len(trackers), ncols=100):
                eao_result.update(ret)

        ar_benchmark.show_result(ar_result, eao_result, show_video_level=args.show_video_level,
                                 log=log)

        with open("VOT2018.pickle", "wb") as output_file:
            pickle.dump(eao_result, output_file)

        if args.vis:
            for attr, videos in dataset.attr.items():
                # print(attr)
                if attr == 'ALL':
                    draw_success_precision(success_ret,
                                           name=dataset.name,
                                           videos=videos,
                                           attr=attr,
                                           precision_ret=precision_ret)

    elif 'VOT2018-LT' == args.dataset:
        dataset = VOTLTDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = F1Benchmark(dataset, model_epoch=args.model_epoch, expcase=args.case)
        f1_result = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval, trackers), desc='eval f1', total=len(trackers), ncols=100):
                f1_result.update(ret)
        benchmark.show_result(f1_result, show_video_level=args.show_video_level)
        return f1_result


if __name__ == '__main__':
    main()
