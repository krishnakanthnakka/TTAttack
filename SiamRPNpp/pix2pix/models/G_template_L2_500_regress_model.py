import torch
import random
import numpy as np
import math
import cv2
np.random.seed(0)
from .base_model import BaseModel
from . import networks
from siamRPNPP import SiamRPNPP
from data_utils import normalize
from utils_TTA import ssim
from torchvision.utils import save_image
cls_thres = 0.7


def get_all_points(opt):

    if opt.directions == 23:

        # HAACK FOR INFERENCE TIME FOR Siam_ocean nd Siam_mobilenet. Please double check for training white box for them.
        if opt.tracker_name in ['siamrpn_r50_l234_dwxcorr', 'siamrpn_r50_l234_dwxcorr_otb', 'siamrpn_r50_l234_dwxcorr_lt',
                                'siamrpn_r50_l234_dwxcorr2', 'siamrpn_r50_l234_dwxcorr_lt2', 'siamban', 'siam_ocean_online', 'siamrpn_mobilev2_l234_dwxcorr']:
            pts_23 = [(4, 0), (-3, -3), (0, -4), (-3, 3), (2, -4), (-4, -2), (-4, 1), (-2, 4), (-1, 4), (3, -3), (4, 2),
                      (3, 3), (1, -4), (2, 4), (0, 4), (-4, 0), (4, -1), (4, -2), (4, 1), (1, 4), (-1, -4), (-4, 2), (-2, -4)]
        return pts_23

    if opt.directions == 12:
        # HAACK FOR INFERENCE TIME FOR Siam_ocean nd Siam_mobilenet. Please double check for training white box for them.
        if opt.tracker_name in ['siamrpn_r50_l234_dwxcorr', 'siamrpn_r50_l234_dwxcorr_otb', 'siamrpn_r50_l234_dwxcorr_lt',
                                'siamrpn_r50_l234_dwxcorr2', 'siamrpn_r50_l234_dwxcorr_lt2', 'siamban', 'siam_ocean_online', 'siamrpn_mobilev2_l234_dwxcorr']:
            pts_12 = [(0, 4), (2, 4), (3, 3), (4, 0), (0, -4), (2, -4),
                      (3, -3), (-2, 4), (-3, 3), (-4, 0), (-2, -4), (-3, -3)]
        return pts_12

    elif opt.directions == 4:

        if opt.tracker_name in ['siamrpn_r50_l234_dwxcorr', 'siamrpn_r50_l234_dwxcorr_otb', 'siamrpn_r50_l234_dwxcorr_lt', 'siamrpn_r50_l234_dwxcorr2',
                                'siamrpn_r50_l234_dwxcorr_lt2', 'siamban', 'siam_ocean_online', 'siamrpn_mobilev2_l234_dwxcorr']:
            pts_4 = [(0, 4), (4, 0), (0, -4), (-4, 0)]

        return pts_4

    elif opt.directions == 8:

        if opt.tracker_name in ['siamrpn_r50_l234_dwxcorr', 'siamrpn_r50_l234_dwxcorr_otb', 'siamrpn_r50_l234_dwxcorr_lt', 'siamrpn_r50_l234_dwxcorr2',
                                'siamrpn_r50_l234_dwxcorr_lt2', 'siamban', 'siam_ocean_online', 'siamrpn_mobilev2_l234_dwxcorr']:
            pts_8 = [(0, 4), (2, 4), (4, 0), (0, -4), (2, -4), (-2, 4), (-4, 0), (-2, -4)]
        return pts_8


def get_center(opt):

    if opt.tracker_name in ['siamrpn_r50_l234_dwxcorr', 'siamrpn_r50_l234_dwxcorr_otb', 'siamrpn_r50_l234_dwxcorr_lt', 'siamrpn_r50_l234_dwxcorr2',
                            'siamrpn_r50_l234_dwxcorr_lt2', 'siamrpn_mobilev2_l234_dwxcorr']:
        return [12, 12], [25, 25]

    if opt.tracker_name in ['siamrpn_alex_dwxcorr_otb', 'siamrpn_alex_dwxcorr', 'DAsiamrpn_alex_dwxcorr_lt', 'DAsiamrpn_alex_dwxcorr']:
        return [8, 8], [17, 17]

    # THIS IS A HACK USED AT INFERENCE
    if opt.tracker_name in ['siam_ocean', 'siam_ocean_online']:
        return [12, 12], [25, 25]

    if opt.tracker_name in ['dimp', 'siamban']:
        return [2, 2], None
    else:
        assert False, 'implement center point function'


class GtemplateL2500regressModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch', netG='unet_128', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='lsgan')
            parser.add_argument('--lambda_L1', type=float, default=500, help='weight for L1 loss')
        return parser

    def __init__(self, opt, log=print):

        BaseModel.__init__(self, opt, log)
        self.loss_names = ['G_L2', 'cls', 'reg', 'cls_T', 'reg_T', 'feat']
        self.visual_names = ['template_clean1', 'template_adv1']
        self.log = log
        if self.isTrain:
            self.model_names = ['G']
        else:
            self.model_names = ['G']

        self.UNTARGETED = not opt.istargeted
        n_inputch = 3 if self.UNTARGETED else 4
        self.netG = networks.define_G(n_inputch, 3, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.criterionL2 = torch.nn.MSELoss()
            self.init_weight_L2 = self.opt.lambda_L1
            self.init_weight_cls = 0.1
            self.init_weight_reg = 1
            self.cls_margin = -5
            self.side_margin1 = -5
            self.side_margin2 = -5
            self.weight_L2 = self.init_weight_L2
            self.weight_cls = self.init_weight_cls
            self.weight_reg = self.init_weight_reg
            self.weight_cls_T = 0.1
            self.weight_reg_T = 100
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

        self.siam = SiamRPNPP(tracker_name=opt.tracker_name, istrain=self.isTrain)

        # since input is [-1, 1] range. We multiply by 2
        self.eps = float(opt.eps * 2) / 255.0
        self.center_point, feature_map_size = get_center(opt)

        if self.UNTARGETED:
            log("Initializing the vanilla Generator without extra channel")
        else:
            log("Initializing the conditional Generator with extra channel")

        log("Center point: {}, Feature Map: {}".format(self.center_point, feature_map_size))
        log("Actual Epsilon: {}".format(self.eps * 255 * 0.5))
        log("Isuntargeted Attack: {}".format(self.UNTARGETED))

        self.maeloss = torch.nn.L1Loss()
        self.index = 1
        self.frame = 1
        self.perturbmetrics = []
        self.is_training = opt.isTrain
        self.universal_perturb = opt.attack_universal
        self.universal_flag = True
        log("Universal perturb: {}".format(self.universal_perturb))

        if not self.UNTARGETED:
            self.TARGETED_ATTACK_RADIUS = 5
            log("Targted attack radius: {}".format(self.TARGETED_ATTACK_RADIUS))

        if self.UNTARGETED and (opt.tracker_name == 'dimp' or opt.tracker_name == 'siam_ocean' or opt.tracker_name == 'siam_ocean_online'):
            return

        self.anchors_box, self.anchors_center = self.siam.generate_all_anchors().all_anchors
        assert self.anchors_box.shape[2] == feature_map_size[0], 'Incorrect anchors selected!'

    def logparams(self, log):
        log("cls_margin   :  {}".format(self.cls_margin))
        log("side_margin1 :  {}".format(self.side_margin1))
        log("side_margin2 :  {}".format(self.side_margin2))
        log("weight_L2    :  {}".format(self.weight_L2))
        log("weight_cls   :  {}".format(self.weight_cls))
        log("weight_reg   :  {}".format(self.weight_reg))
        log("weight_cls_T :  {}".format(self.weight_cls_T))
        log("weight_reg_T :  {}".format(self.weight_reg_T))

    def set_input(self, input):
        self.template_clean255 = input[0].squeeze(0).cuda()
        self.template_clean1 = normalize(self.template_clean255)
        self.X_crops = input[1].squeeze(0).cuda()
        self.search_clean255 = input[1].squeeze(0).cuda()
        self.search_clean1 = normalize(self.search_clean255)
        self.num_search = self.search_clean1.size(0)
        self.shift = input[2][0][0], input[2][1][0]
        self.target_bbox = [0, 0, 10, 10]
        self.folder = input[3]

    def get_mask(self):

        if self.is_training:
            x, y = self.shift
        else:
            dist = 4  # 5 in most initial experiments
            x = int(dist * math.cos(self.dir_))
            y = -int(dist * math.sin(self.dir_))

            # Uncomment below line to get the K=12 diverse perturbations
            #x, y = get_closest_point(x, y)
            # print("get nearest directional perturbation!")

        print("\t\t\t\t Randomx: {}, randomy: {}".format(x, y), end='\r')
        pos = [2, self.center_point[0] + y, self.center_point[1] + x]
        x1, y1, x2, y2 = [int(self.anchors_box[i, pos[0], pos[1], pos[2]] + 127) for i in range(4)]

        self.target_bbox = [(x1 + x2) / 2 - 127, (y1 + y2) / 2 - 127, x2 - x1, y2 - y1]

        mask = np.zeros((255, 255), dtype=np.float32)
        mask[y1:y2, x1:x2] = 1.0
        if 0:
            cv2.imwrite('mask.png', 255 * mask)
        return torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).cuda()

    def forward(self, frame_index, target_sz=(255, 255), dir_=0, enhance=False):

        self.enhance = enhance
        template128_clean = torch.nn.functional.interpolate(
            self.template_clean1, size=(512, 512), mode='bilinear').cuda()

        if self.UNTARGETED:

            if not self.universal_perturb:
                if frame_index == 1:
                    self.perturb = self.netG(template128_clean)
                    perturb = self.perturb.clone()
                    print("calculating the perturbation for first frame")
                else:
                    perturb = self.perturb.clone()
            else:
                if self.universal_flag is True:
                    self.perturb = self.netG(template128_clean)
                    perturb = self.perturb.clone()
                    self.universal_flag = False
                    print("calculating the perturbation for first frame")

                    # torch.save(self.perturb, "./univ.pth")
                    # exit()

                else:
                    perturb = self.perturb.clone()

        else:

            if not self.universal_perturb:
                if frame_index == 1:
                    self.directional_pertub_dict = self.compute_directional_perturbations(template128_clean)
                    perturb = self.get_closest_directional_perturbation(self.directional_pertub_dict, dir_)
                    print("calculating the video dependent directional perturbation at frame {}".format(frame_index))

                else:
                    perturb = self.get_closest_directional_perturbation(self.directional_pertub_dict, dir_)

            else:
                if self.universal_flag is True:
                    self.directional_pertub_dict = self.compute_directional_perturbations(template128_clean)
                    perturb = self.get_closest_directional_perturbation(self.directional_pertub_dict, dir_)
                    self.universal_flag = False
                    print("calculating {}  universal directional perturbations once".format(self.opt.directions))
                    # torch.save(self.directional_pertub_dict, "./dir_univ.pth")
                    # exit()

                else:
                    perturb = self.get_closest_directional_perturbation(
                        self.directional_pertub_dict, dir_).clone().cuda()

        perturb = torch.nn.functional.interpolate(perturb, size=target_sz, mode='bilinear')
        self.search_adv1 = self.search_clean1 + perturb
        self.search_adv1 = torch.min(
            torch.max(self.search_adv1, self.search_clean1 - self.eps), self.search_clean1 + self.eps)
        self.search_adv1 = torch.clamp(self.search_adv1, -1.0, 1.0)
        self.search_adv255 = self.search_adv1 * 127.5 + 127.5
        self.frame += 1

        # dummy metrics
        return {"target_bbox": [0, 0, 10, 10], "metrics": {"MAE": torch.tensor(0.0), "Linf": self.eps / 2, "SSIM": 100.0}}

    def compute_directional_perturbations(self, template128_clean):

        pts_x_y_all = get_all_points(self.opt)
        directional_pertub_dict = {}

        for index, point in enumerate(pts_x_y_all):

            x, y = point
            pos = [2, self.center_point[0] + y, self.center_point[1] + x]
            x1, y1, x2, y2 = [int(self.anchors_box[i, pos[0], pos[1], pos[2]] + 127) for i in range(4)]
            mask = np.zeros((255, 255), dtype=np.float32)
            mask[y1:y2, x1:x2] = 1.0
            mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).cuda()
            mask = torch.nn.functional.interpolate(mask, size=(512, 512), mode='nearest')
            perturb = self.netG(torch.cat((template128_clean, mask), dim=1))
            directional_pertub_dict[str(index)] = perturb.clone().cuda()

        return directional_pertub_dict

    def get_closest_directional_perturbation(self, directional_pertub_dict, dir_):

        dist = self.TARGETED_ATTACK_RADIUS
        x = int(dist * math.cos(dir_))
        y = -int(dist * math.sin(dir_))
        test_node = (x, y)
        pts_x_y_all = get_all_points(self.opt)
        nodes = np.asarray(pts_x_y_all)
        dist_2 = np.sum((nodes - test_node)**2, axis=1)
        index = np.argmin(dist_2)
        return directional_pertub_dict[str(index)]

    def visualize(self, target_sz=(255, 255)):

        self.siam.model.template(self.template_clean255)
        self.template_clean1 = normalize(self.template_clean255)

        template128_clean = torch.nn.functional.interpolate(
            self.template_clean1, size=(512, 512), mode='bilinear').cuda()

        if self.UNTARGETED:
            perturb = self.netG(template128_clean)
        else:
            mask = self.get_mask()
            mask = torch.nn.functional.interpolate(mask, size=(512, 512), mode='nearest')
            perturb = self.netG(torch.cat((template128_clean, mask), dim=1))

        perturb = torch.nn.functional.interpolate(perturb, size=target_sz, mode='bilinear')
        search_adv1 = self.search_clean1 + perturb
        search_adv1 = torch.min(torch.max(search_adv1, self.search_clean1 - self.eps),
                                self.search_clean1 + self.eps)
        search_adv1 = torch.clamp(search_adv1, -1.0, 1.0)
        search_adv255 = search_adv1 * 127.5 + 127.5
        temp = self.search_clean1 * 127.5 + 127.5
        self.siam.visualize(search_adv255, self.index)
        loss = torch.nn.L1Loss()
        print("VMAE:{:.4f}, L-inf Norm:{:.4f}".format(loss(search_adv255, temp),
                                                      torch.max(torch.abs(search_adv255 - temp))), end="\r")
        self.index += 1

    def backward_G(self):

        self.loss_G_L2, self.loss_cls, self.loss_reg, self.loss_cls_T, self.loss_reg_T, self.loss_feat = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        self.loss_G_L2 = self.criterionL2(self.search_adv1, self.search_clean1) * self.weight_L2
        attention_mask = (self.score_maps_clean > cls_thres)
        num_attention = int(torch.sum(attention_mask))

        # code for extracting target, makee sure to pass clean image
        # by mulitplying with zero perturbation in forward
        # print(torch.max(self.score_maps_clean))
        # self.loss_feat = self.siam.get_feat_loss()

        if num_attention > 0:

            if 1:
                score_map_adv_att = self.score_maps_adv[attention_mask]
                reg_adv_att = self.reg_res_adv[2:4, attention_mask]
                self.loss_cls = torch.mean(torch.clamp(
                    score_map_adv_att[:, 1] - score_map_adv_att[:, 0], min=self.cls_margin)) * self.weight_cls
                self.loss_reg = (torch.mean(torch.clamp(reg_adv_att[0, :], min=self.side_margin1)) +
                                 torch.mean(torch.clamp(reg_adv_att[1, :], min=self.side_margin2))) * self.weight_reg
            # combine loss and calculate gradients
            self.loss_G = self.loss_G_L2 + self.loss_cls + self.loss_reg

            # -------------------------------- CHANGED --------------------------------  No. 1
            if 1:

                self.loss_cls_T, self.loss_reg_T = self.siam.get_target_cls_reg(
                    self.search_adv255, self.shift, self.center_point)
                # (5HWN,2)without softmax,(5HWN,4)
                # self.loss_cls_T, self.loss_reg_T = self.siam.get_target_cls_reg_adaptive(self.search_adv255)  # (5HWN,2)without softmax,(5HWN,4)

                self.loss_cls_T *= self.weight_cls_T
                self.loss_reg_T *= self.weight_reg_T
                self.loss_G += self.loss_cls_T + self.loss_reg_T

            if 0:
                self.loss_feat = self.siam.get_feat_loss() * 10
                self.loss_G += self.loss_feat

        else:
            self.loss_G = self.loss_G_L2
        self.loss_G.backward()

    def optimize_parameters(self):

        with torch.no_grad():
            self.siam.model.template(self.template_clean255)
            self.score_maps_clean = self.siam.get_heat_map(self.X_crops, softmax=True)

        self.forward()
        self.score_maps_adv, self.reg_res_adv = self.siam.get_cls_reg(self.search_adv255, softmax=False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
