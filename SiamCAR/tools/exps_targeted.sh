
#OTB100
#python test_attack_ours_target.py  --dataset OTB100  --snapshot ../../tracker_weights/siamcar_general/model_general.pth  --model_iter=4_net_G.pth --case=2 --eps=16 --attack_universal --trajcase=SE  --vis



# VOT2018
python test_attack_ours_target.py  --dataset VOT2018  --snapshot ../../tracker_weights/siamcar_general/model_general.pth  --model_iter=4_net_G.pth --case=2 --eps=16 --attack_universal --trajcase=SE
