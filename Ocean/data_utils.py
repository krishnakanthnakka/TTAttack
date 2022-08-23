import torch
from torch.utils.data.dataset import Dataset
import os
import glob
import cv2
import numpy as np
from common_path import train_set_path_ as dataset_dir
import random
from shutil import copy
import datetime
import math
from pysot.core.config import cfg
from pysot.utils.anchor import Anchors


def img2tensor(img_arr):
    '''float64 ndarray (H,W,3) ---> float32 torch tensor (1,3,H,W)'''
    img_arr = img_arr.astype(np.float32)
    img_arr = img_arr.transpose(2, 0, 1)  # channel first
    img_arr = img_arr[np.newaxis, :, :, :]
    init_tensor = torch.from_numpy(img_arr)  # (1,3,H,W)
    return init_tensor


def normalize(im_tensor):
    '''(0,255) ---> (-1,1)'''
    im_tensor = im_tensor / 255.0
    im_tensor = im_tensor - 0.5
    im_tensor = im_tensor / 0.5
    return im_tensor


def tensor2img(tensor):
    '''(0,255) tensor ---> (0,255) img'''
    '''(1,3,H,W) ---> (H,W,3)'''
    tensor = tensor.squeeze(0).permute(1, 2, 0)
    img = tensor.cpu().numpy().clip(0, 255).astype(np.uint8)
    return img


def shift_image_up(search_arr, l, r, u, d):

    search_image = np.zeros(search_arr.shape, search_arr.dtype)
    channel_average = np.mean(search_arr, axis=(0, 1))
    search_image = search_image + channel_average
    # print(u)
    search_image[:-u] = search_arr[u:]
    return search_image


def shift_image_down(search_arr, l, r, u, d):

    search_image = np.zeros(search_arr.shape, search_arr.dtype)
    channel_average = np.mean(search_arr, axis=(0, 1))
    search_image = search_image + channel_average
    # print(u)
    search_image[d:] = search_arr[:-d]
    return search_image


def shift_image_right(search_arr, l, r, u, d):

    search_image = np.zeros(search_arr.shape, search_arr.dtype)
    channel_average = np.mean(search_arr, axis=(0, 1))
    search_image = search_image + channel_average
    # print(u)
    search_image[:, r:] = search_arr[:, :-r]
    return search_image


def shift_image_left(search_arr, l, r, u, d):

    search_image = np.zeros(search_arr.shape, search_arr.dtype)
    channel_average = np.mean(search_arr, axis=(0, 1))
    search_image = search_image + channel_average
    # print(u)
    search_image[:, :-l] = search_arr[:, l:]
    return search_image


def generate_all_anchors():
    anchors2 = Anchors(cfg.ANCHOR.STRIDE, cfg.ANCHOR.RATIOS, cfg.ANCHOR.SCALES)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ CHECK THIS and radius ++++++++++++++++++++++++++++++++++++++++++

    anchors2.generate_all_anchors(im_c=0, size=cfg.TRACK.OUTPUT_SIZE)
    #anchors2.generate_all_anchors(im_c=0, size=cfg.TRAIN.OUTPUT_SIZE)
    return anchors2


def get_shift(r):

    circle_x = 0
    circle_y = 0
    a = random.random() * 2 * math.pi
    x = int(r * math.cos(a) + circle_x)
    y = int(r * math.sin(a) + circle_y)

    return (x, y)


def shift_image_random(search_arr, dist, shiftdir):

    search_image = np.zeros(search_arr.shape, search_arr.dtype)
    channel_average = np.mean(search_arr, axis=(0, 1))
    search_image = search_image + channel_average

    x, y = shiftdir

    dir_ = (math.atan2(-y, x) + 2 * math.pi) % (2 * math.pi)
    a = (dir_ + math.pi) % (2 * math.pi)

    #dist = 6
    x = int(dist * math.cos(a))
    y = int(dist * math.sin(a))

    #print(dist, x, y)

    #print(x, y, dist)

    #print("Augmentation:", x, y, dist)

    # RIGHT
    if(x > 0):
        #print('shift right')
        search_image[:, x:] = search_arr[:, :-x]
    elif x < 0:
        #print('shift left')
        x = abs(x)
        search_image[:, :-x] = search_arr[:, x:]

    # CHANGED. ADDED RECENTLY IN MARCH
    # elif x == 0:
    #     search_image = search_arr + 0

    # UP
    if(y > 0):
        #print('shift up')
        # WRONG BUG.. it was search_image before
        search_image[:-y] = search_image[y:]
    elif y < 0:
        #print('shift down')

        # WRONG BUG.. it was search_image before
        y = abs(y)
        search_image[y:] = search_image[:-y]

    return search_image


def shift_image_random2(search_arr):

    prob = random.uniform(0, 1)

    if random.uniform(0, 1) < 0.6:
        search_arr = shift_image_up(search_arr, l=0, r=0, u=random.randint(10, 110), d=0)
        shift = (0, 4)

    # elif random.uniform(0, 1) < 0.5:
    #     search_arr = shift_image_down(search_arr, l=0, r=0, u=0, d=random.randint(10, 110))
    #     shift = (0, -4)

    elif random.uniform(0, 1) < 0.66:
        search_arr = shift_image_right(search_arr, l=0, r=random.randint(10, 110), u=0, d=0)
        shift = (-4, 0)

    elif random.uniform(0, 1) < 0.99:
        search_arr = shift_image_left(search_arr, l=random.randint(10, 110), r=0, u=0, d=0)
        shift = (4, 0)

    return search_arr, shift


class GOT10k_dataset(Dataset):
    def __init__(self, max_num=15, opt=None):
        folders = sorted(os.listdir(dataset_dir))
        folders.remove('init_gt.txt')
        self.folders_list = [os.path.join(dataset_dir, folder) for folder in folders]
        self.max_num = max_num
        self.anchors = generate_all_anchors()
        self.opt = opt

    def __getitem__(self, index):
        cur_folder = self.folders_list[index]
        img_paths = sorted(glob.glob(os.path.join(cur_folder, '*.jpg')))
        '''get init frame tensor'''
        init_frame_path = img_paths[0]
        init_frame_arr = cv2.imread(init_frame_path)
        init_tensor = img2tensor(init_frame_arr)
        '''get search regions' tensor'''
        search_region_paths = img_paths[1:self.max_num + 1]  # to avoid being out of GPU memory
        num_search = len(search_region_paths)

        #num_search = 1

        search_tensor = torch.zeros((num_search, 3, 255, 255), dtype=torch.float32)

       # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ CHECK THIS and radius ++++++++++++++++++++++++++++++++++++++++++

        if self.opt.tracker_name in ['siamrpn_r50_l234_dwxcorr', 'siamrpn_r50_l234_dwxcorr_otb', 'siamrpn_r50_l234_dwxcorr_lt', 'siamrpn_r50_l234_dwxcorr2', ' siamrpn_r50_l234_dwxcorr_lt2']:
            radius = 4
        elif self.opt.tracker_name in ['siamrpn_alex_dwxcorr_otb', 'siamrpn_alex_dwxcorr', 'DAsiamrpn_alex_dwxcorr_lt']:
            radius = 3

        #print(" Attack target radius :{}\n ".format(radius))

        # shift = (-radius, 0)  # for drift left
        # shift = (0, -radius)  # for drift up

        if self.opt.istargeted:

            # CHANGED FOR ON EXP
            shift = get_shift(r=radius)
            #shift = (0, 3)
            # print(shift)

        for i in range(num_search):
            search_arr = cv2.imread(search_region_paths[i])
            # print(shift)

            # DRFIT DOWN
            if not self.opt.istargeted:

                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ CHECK THIS and radius ++++++++++++++++++++++++++++++++++++++++++

                shift = (0, radius)  # for drift down
                #shift = get_shift(r=radius)

                if random.uniform(0, 1) > 0.5:
                    search_arr = shift_image_up(search_arr, l=0, r=0, u=random.randint(10, 75), d=0)

            # # DRFIT UP
            # if random.uniform(0, 1) > 0.5:
            #     search_arr = shift_image_down(search_arr, l=0, r=0, u=0, d=random.randint(10, 75))

            # DRIFT LEFT
            # if random.uniform(0, 1) > 0.5:
            #     search_arr = shift_image_right(search_arr, l=0, r=random.randint(10, 75), u=0, d=0)

            #  # DRIFT RIGHTT
            # if random.uniform(0, 1) > 0.5:
            #     search_arr = shift_image_left(search_arr, l=random.randint(10, 75), r=0, u=0, d=0)

            # FOR TARGTED ATTACKS

            if self.opt.istargeted:
                if random.uniform(0, 1) > 0.5:
                    search_arr = shift_image_random(search_arr, random.randint(15, 80), shift)

            # elif self.opt.istargeted:
            #     #shift = get_shift(r=radius)
            #     #print(shift)
            #     if random.uniform(0, 1) > 0.3:
            #         search_arr, shift = shift_image_random2(search_arr)

            search_tensor[i] = img2tensor(search_arr)

        '''Note: we don't normalize these tensors here,
        but leave normalization to training process'''
        # print(shift)
        # exit()
        return (init_tensor, search_tensor, shift, cur_folder)

    def __len__(self):
        return len(self.folders_list)


def savefiles(opt):
    dst_folder = os.path.join(opt.checkpoints_dir, opt.name + "_" + str(opt.case),
                              "files_{}".format(datetime.datetime.now().strftime("%H_%M_%S")))

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    copy("./train1.py", dst_folder)
    copy("./pix2pix/models/G_template_L2_500_regress_model.py", dst_folder)
    copy("./data_utils.py", dst_folder)
    copy("./siamRPNPP.py", dst_folder)
    copy("./pysot/pysot/tracker/siamrpn_tracker.py", dst_folder)
    copy("./pix2pix/options/train_options1.py", dst_folder)
    copy("./pix2pix/options/base_options1.py", dst_folder)
