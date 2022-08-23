import os
'''Things that we can change'''
###################################################

siam_model_ = ''  # 'siamrpn_alex_dwxcorr_otb'
#siam_model_ = 'siamrpn_r50_l234_dwxcorr_otb'


###################################################
dataset_name_ = ''  # 'OTB100'
# dataset_name_ = 'VOT2018'
# dataset_name_ = 'LaSOT'
##################################################
# video_name_ = 'CarScale' # worser(inaccurate scale estimation)
# video_name_ = 'Bolt' # fail earlier(distractor)
# video_name_ = 'Doll' # unstable
# video_name_ = 'ants1'
# video_name_ = 'airplane-1'
video_name_ = ''
#########################################################################################
'''change to yours'''

import os
root_dir = os.path.dirname(os.path.abspath(__file__))

project_path_ = root_dir
print(root_dir)


dataset_root_ = os.path.join(root_dir, 'data')  # make links for used datasets

#train_set_path_ = '/cvlabdata1/home/krishna/AttTracker/GOT/train/'
train_set_path_ = os.path.join(root_dir, 'data/GOT/train')  # './data/GOT/train'
