
""" Code Usage:
python train1.py --eps=8 --case=252  --tracker_name=siamrpn_r50_l234_dwxcorr


python train1.py --eps=8 --case=300  --tracker_name=siamrpn_alex_dwxcorr


"""

import os

import datetime
import torch
import time

from data_utils import GOT10k_dataset, savefiles
from models import create_model
from pysot.utils.distributed import (dist_init, DistModule, reduce_gradients,
                                     average_reduce, get_rank, get_world_size)
from options.train_options1 import TrainOptions
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from util.visualizer import Visualizer
from utils.log import create_logger

gpu = '0'
MULTI_GPU = True if gpu in ['0,1', '1,0'] else False
os.environ['CUDA_VISIBLE_DEVICES'] = gpu


if __name__ == '__main__':

    log, logclose = create_logger(log_filename=os.path.join('./logs/', 'trainlog_{}.txt'.
                                                            format(datetime.datetime.now().strftime("%H:%M:%S"))))
    opt = TrainOptions().parse(log)
    model = create_model(opt)
    model.setup(opt)
    model.logparams(log)
    savefiles(opt)

    if MULTI_GPU:
        distmodel = torch.nn.DataParallel(model)
    else:
        distmodel = model

    dataset = GOT10k_dataset(max_num=15, opt=opt)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)  # CHANGED SHUFFL
    dataset_size = len(dataloader)
    log('The number of training images = %d' % dataset_size)
    total_iters = 0

    for epoch in range(opt.epoch_count, opt.epoch_count + 200):

        log("---------------- Epoch: {} ------------------------".format(epoch))
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        if MULTI_GPU:
            distmodel.module.index = 1
        else:
            distmodel.index = 1

        iter_st_time = time.time()

        for i, data in enumerate(dataloader):

            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            if MULTI_GPU:
                distmodel.module.set_input(data)
            else:
                distmodel.set_input(data)

            if MULTI_GPU:
                distmodel.module.optimize_parameters()
            else:
                distmodel.optimize_parameters()

            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                if MULTI_GPU:
                    distmodel.module.visualize()
                else:
                    distmodel.visualize()

            # exit()

            if total_iters % opt.print_freq == 0:
                if MULTI_GPU:
                    losses = distmodel.module.get_current_losses()
                else:
                    losses = distmodel.get_current_losses()

                t_comp = (time.time() - iter_st_time) / opt.batch_size

                log("Time : {:5d}, epoch : {:3d},  iter : {:4d}/{:4d}, l2 : {:6.3f}, cls : {:6.3f}, \
                    reg : {:.3f}, cls_T : {:.3f}, reg_T : {:.3f}, feat : {:.3f}".format(int(t_comp), epoch, i,
                                                                                        len(dataloader),
                                                                                        losses['G_L2'],
                                                                                        losses['cls'],
                                                                                        losses['reg'],
                                                                                        losses['cls_T'],
                                                                                        losses['reg_T'],
                                                                                        losses['feat']))

            if total_iters % opt.save_latest_freq == 0:
                log('saving the latest model (epoch %d, total_iters %d)' %
                    (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'

                if MULTI_GPU:
                    distmodel.module.save_networks(save_suffix)
                else:
                    distmodel.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            log('saving the model at the end of epoch %d, iters %d' %
                (epoch, total_iters))
            if MULTI_GPU:
                distmodel.module.save_networks('latest')
                distmodel.module.save_networks(epoch)
            else:
                distmodel.save_networks('latest')
                distmodel.save_networks(epoch)

        log('End of epoch %d / %d \t Time Taken: %d sec' %
            (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        if MULTI_GPU:
            distmodel.module.update_learning_rate()
        else:
            distmodel.update_learning_rate()

    print("Completed training!")
