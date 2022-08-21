import os
import sys

from common_path import project_path_
from models import create_model
from options.test_options0 import TestOptions

opt = TestOptions().parse()

opt.model = 'G_template_L2_500_regress'
opt.netG = 'unet_128'

sys.path.insert(0, os.path.join(project_path_, 'checkpoints/{}_{}/'.format(opt.model, opt.case)))
print("Path: {}".format(os.path.join(project_path_, 'checkpoints/{}_{}/'.format(opt.model, opt.case))))
from config_new import eps, istargeted
opt.eps = eps
opt.istargeted = istargeted
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu


# # --- Create Model
# GAN = create_model(opt)
# model_epoch = opt.model_iter
# expcase = opt.case
# GAN.load_path = os.path.join(project_path_, 'checkpoints/{}_{}/{}'.format(opt.model, expcase, model_epoch))
# GAN.setup(opt)
# GAN.eval()


def get_model_GAN(log):
    GAN = create_model(opt, log)
    model_epoch = opt.model_iter
    expcase = opt.case
    GAN.load_path = os.path.join(
        project_path_, 'checkpoints/{}_{}/{}'.format(opt.model, expcase, model_epoch))
    GAN.setup(opt)
    GAN.eval()
    return GAN, opt
