## Code is in early release. Will be updated shortly


## Introduction

This is an official release of the paper **Universal, Transferable Adversarial Perturbations for Visual Object Trackers**.
![images](./images/frameworkfull_new_V2.png)

**Abstract.** In recent years, Siamese networks have led to great progress in visual object tracking. While these methods were shown to be vulnerable to adversarial attacks, the existing attack strategies do not truly pose great practical threats. They either are too expensive to be performed online, require computing image-dependent perturbations, lead to unrealistic trajectories, or suffer from weak transferability to other black-box trackers. In this paper, we address the above limitations by showing the existence of a universal perturbation that is image agnostic and fools black-box trackers at virtually no cost of perturbation. Furthermore, we show that our framework can be extended to the challenging targeted attack setting that forces the tracker to follow any given trajectory by using diverse directional universal perturbations. At the core of our framework, we propose to learn to generate a single perturbation from the object template only, that can be added to every search image and still successfully fool the tracker for the entire video. As a consequence, the resulting generator outputs perturbations that are quasi-independent of the template, thereby making them universal perturbations. Our extensive experiments on four benchmarks datasets, i.e., OTB100, VOT2019, UAV123, and LaSOT, demonstrate that our universal transferable per- turbations (computed on SiamRPN++) are highly effective when transferred to other state-of-the-art trackers, such as SiamBAN, SiamCAR, DiMP, and Ocean online.



## Installation

1. It is tested with the following packages and hardware:

    ``` text
      PyTorch: 1.5.0
      Python: 3.6.9
      Torchvision: 0.6.0
      CUDA: 10.1
      CUDNN: 7603
      NumPy: 1.18.1
      PIL: 7.0.0
      GPU: Tesla V100-SXM2-32GB
   ```

2. Download source code from GitHub
   ```
    git clone https://github.com/krishnakanthnakka/TTAttack.git
   ```

3. We release  the pretrained perturbation generators against SiamRPN++ (R) on [GoogleDrive](https://drive.google.com/drive/folders/1CawsQuwFiGlHxqLYM2BOOF9NxxI-U_m4?usp=sharing).
Please place the ```checkpoints``` folder inside the  ```./SiamRPNpp/``` folder.




## Setting the datasets and tracker checkpoints

1. Please download the weights of different trackers on [GoogleDrive](https://drive.google.com/drive/folders/1CawsQuwFiGlHxqLYM2BOOF9NxxI-U_m4?usp=sharing). These are taken from the original repositories of the respective papers.
  Place the downloaded ```tracker_weights``` folder in the root folder

2. Create a folder named ```testing_dataset``` in  the root  folder and place all the datasets such as ```OTB100```, ```VOT2018```, ```UAV123```, ```LASOT``` inside it.The dataset should be organized as:
    ```
    testing_dataset
     ├── OTB100
     ├── VOT2018
     ├── UAV123
     ├── lasot
    ```






## Untargeted attack on SiamRPN++ (M)

1. Enter the directory of ```SiamRPNpp``` tracker by ```cd SiamRPNpp```

2. Set all environmental paths and other packages in path by ```source envs.sh```

3. For attacking SiamRPN++ (M) tracker using the generator trained on ```SiamRPN++ (R)``` as discriminator and ```GOT10K``` dataset:

   ```py

    cd pysot/tools

    # universal attack (Ours) on  OTB100 with  SiamRPN++ (M) tracker
    python tta_attack.py   --tracker_name=siamrpn_mobilev2_l234_dwxcorr --dataset=OTB100 --case=1 --gpu=1 --model_iter=4_net_G.pth --attack_universal

    # template dependent attack (Ours (TD)) on  OTB100 with  SiamRPN++ (M) tracker
    python tta_attack.py   --tracker_name=siamrpn_mobilev2_l234_dwxcorr --dataset=OTB100 --case=1 --gpu=1 --model_iter=4_net_G.pth
   ```

#### Results on SiamRPN++ (M) tracker

4. We observe the following results as in Table 1.
    | Method | Success  | Precision |
    | :---:  | :---:     | :---:    |
    |  Normal | 0.657     | 0.862    |
    |  Ours (TD)  | 0.217  | 0.281    |
    |  Ours  | 0.212     | 0.275     |


## Untargeted attack on SiamBAN

1. Enter the directory of ```SiamBAN``` tracker by ```cd SiamBAN```

2. Set all environmental paths and other packages in path by ```source envs.sh```

3. For attacking SiamBAN tracker using the generator trained on ```SiamRPN++ (R)``` as discriminator and ```GOT10K``` dataset:

   ```py

    # universal attack (Ours)  on  OTB100 with SiamBAN tracker
    cd experiments/siamban_r50_l234_otb
    python -u ../../tools/test_attack_ours.py --snapshot ../../../tracker_weights/siamban_r50_l234_otb/model.pth  --dataset OTB100 --config ../../../tracker_weights/siamban_r50_l234_otb/config.yaml  --model_iter=4_net_G.pth --case=1 --eps=8  --attack_universal


    # template dependent attack (Ours (TD))  on  OTB100 with SiamBAN tracker
    cd experiments/siamban_r50_l234_otb
    python -u ../../tools/test_attack_ours.py --snapshot ../../../tracker_weights/siamban_r50_l234_otb/model.pth  --dataset OTB100 --config ../../../tracker_weights/siamban_r50_l234_otb/config.yaml  --model_iter=4_net_G.pth --case=1 --eps=8

   ```

## Untargeted attack on SiamCAR

1. Enter the directory of ```SiamCAR``` tracker by ```cd SiamCAR```

2. Set all environmental paths and other packages in path by ```source envs.sh```

3. For attacking SiamCAR tracker using the generator trained on ```SiamRPN++ (R)``` as discriminator and ```GOT10K``` dataset:

   ```py

    cd tools

    # universal attack (Ours) on  OTB100 with SiamCAR tracker
    python test_attack_ours.py  --dataset OTB100  --snapshot ../../tracker_weights/siamcar_general/model_general.pth   --model_iter=4_net_G.pth --case=1 --eps=8  --attack_universal


    # template dependent attack (Ours (TD)) on  OTB100 with SiamCAR tracker
    python test_attack_ours.py  --dataset OTB100  --snapshot ../../tracker_weights/siamcar_general/model_general.pth   --model_iter=4_net_G.pth --case=1 --eps=8

   ```


## Untargeted attack on Ocean (Online)

1. Enter the directory of ```Ocean``` tracker by ```cd Ocean```

2. Set all environmental paths and other packages in path by ```source envs.sh```

3. For attacking Ocean (online) tracker using the generator trained on ```SiamRPN++ (R)``` as discriminator and ```GOT10K``` dataset:

   ```py

    cd tools

    # universal attack (Ours) on  OTB100 with Ocean tracker
    python ttattack_untargeted.py --dataset=OTB100  --tracker_name=siam_ocean_online --case=1  --model_iter=4_net_G.pth --gpu=0  --attack_universal


    # template dependent attack (Ours (TD)) on  OTB100 with Ocean tracker
    python ttattack_untargeted.py --dataset=OTB100  --tracker_name=siam_ocean_online --case=1  --model_iter=4_net_G.pth --gpu=0

   ```

