
import json
import os
import numpy as np

from .dataset import Dataset
from .video import Video
from glob import glob
from PIL import Image
from tqdm import tqdm


class OTBVideo(Video):
    def __init__(self, name, root, video_dir, init_rect,
                 img_names, gt_rect, attr, load_img=False):

        super(OTBVideo, self).__init__(name, root, video_dir, init_rect,
                                       img_names, gt_rect, attr, load_img)

    def load_tracker(self, path, tracker_names=None, store=True,
                     model_epoch=None, expcase=None, trajcase=None):

        if not tracker_names:
            tracker_names = [x.split('/')[-1] for x in glob(path) if os.path.isdir(x)]

        if isinstance(tracker_names, str):
            tracker_names = [tracker_names]

        for name in tracker_names:

            if trajcase is None:
                traj_file = os.path.join(path, name, str(expcase), model_epoch, self.name + '.txt')
            else:
                traj_file = os.path.join(path, name, str(expcase), model_epoch, str(trajcase),
                                         self.name + '.txt')

            #print("YOYO", traj_file)
            # exit()

            #traj_file = os.path.join(path, name, str(expcase), model_epoch, self.name, self.name + '_001.txt')
            #print("DEBUG: ", traj_file)
            if not os.path.exists(traj_file):
                if self.name == 'FleetFace':
                    txt_name = 'fleetface.txt'
                elif self.name == 'Jogging-1':
                    txt_name = 'jogging_1.txt'
                elif self.name == 'Jogging-2':
                    txt_name = 'jogging_2.txt'
                elif self.name == 'Skating2-1':
                    txt_name = 'skating2_1.txt'
                elif self.name == 'Skating2-2':
                    txt_name = 'skating2_2.txt'
                elif self.name == 'FaceOcc1':
                    txt_name = 'faceocc1.txt'
                elif self.name == 'FaceOcc2':
                    txt_name = 'faceocc2.txt'
                elif self.name == 'Human4-2':
                    txt_name = 'human4_2.txt'
                else:
                    txt_name = self.name[0].lower() + self.name[1:] + '.txt'
                traj_file = os.path.join(path, name, txt_name)

            if os.path.exists(traj_file):
                with open(traj_file, 'r') as f:

                    #pred_traj = [list(map(float, x.strip().split(','))) for x in f.readlines()]
                    pred_traj = [list(map(float, x.strip().split(','))) for x in f.readlines()]




                    #pred_traj[0] = pred_traj[1]
                    if len(pred_traj) != len(self.gt_traj):
                        print(name, len(pred_traj), len(self.gt_traj), self.name)
                    if store:
                        self.pred_trajs[name] = pred_traj
                    else:
                        return pred_traj
            else:
                print("File Do Not Exists! ", traj_file)

        self.tracker_names = list(self.pred_trajs.keys())
        return


class OTBDataset(Dataset):
    def __init__(self, name, dataset_root, load_img=False):
        super(OTBDataset, self).__init__(name, dataset_root)
        with open(os.path.join(dataset_root, name + '.json'), 'r') as f:
            meta_data = json.load(f)

        # Load videos
        pbar = tqdm(meta_data.keys(), desc='loading ' + name, ncols=100)
        self.videos = {}
        for video in pbar:
            pbar.set_postfix_str(video)
            self.videos[video] = OTBVideo(video,
                                          dataset_root,
                                          meta_data[video]['video_dir'],
                                          meta_data[video]['init_rect'],
                                          meta_data[video]['img_names'],
                                          meta_data[video]['gt_rect'],
                                          meta_data[video]['attr'],
                                          load_img)
        attr = []
        for x in self.videos.values():
            attr += x.attr
        attr = set(attr)
        self.attr = {}
        self.attr['ALL'] = list(self.videos.keys())
        for x in attr:
            self.attr[x] = []
        for k, v in self.videos.items():
            for attr_ in v.attr:
                self.attr[attr_].append(k)
