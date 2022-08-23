import cv2
import numpy as np
import torch





def normalize(im_tensor):
    '''(0,255) ---> (-1,1)'''
    im_tensor = im_tensor / 255.0
    im_tensor = im_tensor - 0.5
    im_tensor = im_tensor / 0.5
    return im_tensor




def adv_attack_template(img_tensor, GAN):
    '''adversarial attack to template'''
    '''input: pytorch tensor(0,255) ---> output: pytorch tensor(0,255)'''
    '''step1: Normalization'''
    img_tensor = normalize(img_tensor)
    '''step2: pass to G'''
    with torch.no_grad():
        GAN.template_clean1 = img_tensor
        perturb_metrics = GAN.forward()
    img_adv = GAN.template_adv255
    return img_adv, perturb_metrics


def adv_attack_template_S(img_tensor, GAN, target_sz=(127, 127)):
    '''adversarial attack to template'''
    '''input: pytorch tensor(0,255) ---> output: pytorch tensor(0,255)'''
    '''step1: Normalization'''
    img_tensor = normalize(img_tensor)
    '''step2: pass to G'''
    with torch.no_grad():
        img_adv = GAN.transform(img_tensor, target_sz)
        return img_adv, {}


def adv_attack_search(img_tensor, GAN, search_sz=(255, 255)):
    '''adversarial attack to search region'''
    '''input: pytorch tensor(0,255) ---> output: pytorch tensor(0,255)'''
    '''step1: Normalization'''
    img_tensor = normalize(img_tensor)
    '''step2: pass to G'''
    with torch.no_grad():
        GAN.search_clean1 = img_tensor
        GAN.num_search = img_tensor.size(0)
        perturb_metrics = GAN.forward(search_sz)
        # print(perturb_metrics)
    img_adv = GAN.search_adv255.clone()
    del GAN.search_adv255,  GAN.search_clean1
    return img_adv, perturb_metrics






def adv_attack_searchtemplate(img_tensor, img_tensor2, GAN, search_sz=(255, 255)):
    '''adversarial attack to search region'''
    '''input: pytorch tensor(0,255) ---> output: pytorch tensor(0,255)'''
    '''step1: Normalization'''
    img_tensor = normalize(img_tensor)
    img_tensor2 = normalize(img_tensor2)

    with torch.no_grad():
        GAN.search_clean1 = img_tensor
        GAN.template_clean1 = img_tensor2
        GAN.num_search = img_tensor.size(0)
        GAN.forward(search_sz)
    img_adv = GAN.search_adv255
    return img_adv


def adv_attack_search_T(img_tensor, img_tensor2, GAN, dir_, idx, search_sz=(255, 255), enhance=False):
    '''adversarial attack to search region'''
    '''input: pytorch tensor(0,255) ---> output: pytorch tensor(0,255)'''
    '''step1: Normalization'''

    # print(torch.max(img_tensor), torch.min(img_tensor))

    img_clean = img_tensor
    img_tensor = normalize(img_tensor)
    img_tensor2 = normalize(img_tensor2)

    # print(torch.max(img_tensor2))

    with torch.no_grad():
        GAN.search_clean1 = img_tensor
        GAN.template_clean1 = img_tensor2
        GAN.num_search = img_tensor.size(0)
        perturbmetrics = GAN.forward(idx, search_sz, dir_, enhance)

    img_adv = GAN.search_adv255
    err = torch.abs(img_clean - img_adv)
    return img_adv, perturbmetrics


def adv_attack_search_new(img_tensor, GAN, search_sz=(255, 255)):
    '''adversarial attack to search region'''
    '''input: pytorch tensor(0,255) ---> output: pytorch tensor(0,255)'''
    '''step1: Normalization'''

    img_tensor = normalize(img_tensor)
    with torch.no_grad():
        GAN.tensor_clean1 = img_tensor
        GAN.num_search = img_tensor.size(0)
        GAN.forward(search_sz)

    img_adv = GAN.tensor_adv255
    return img_adv


def add_gauss_noise(input, sigma):
    gauss_noise = torch.randn(input.size()) * (sigma * 255)
    gauss_noise = gauss_noise.cuda()
    output = input + gauss_noise
    output = output.clamp(0, 255)
    return output


def generate_impulse_mask(im_sz, prob):
    rdn = torch.rand(im_sz)
    mask0 = rdn < (prob / 2)
    mask1 = rdn > (1 - prob / 2)
    return mask0.cuda(), mask1.cuda()


def add_pulse_noise(input, prob):
    mask0, mask1 = generate_impulse_mask(input.size(), prob)
    output = input.clone()
    output[mask0] = 0
    output[mask1] = 255
    return output
