
# OTB100
#cd experiments/siamban_r50_l234_otb
#python -u ../../tools/test_attack_ours_targeted.py --snapshot ../../../tracker_weights/siamban_r50_l234_otb/model.pth  --dataset OTB100     --config ../../../tracker_weights/siamban_r50_l234_otb/config.yaml  --model_iter=4_net_G.pth --case=2 --eps=16  --trajcase=SE   --attack_universal --istargeted




# VOT2018
cd experiments/siamban_r50_l234_otb
python -u ../../tools/test_attack_ours_targeted.py --snapshot ../../../tracker_weights/siamban_r50_l234_otb/model.pth  --dataset VOT2018     --config ../../../tracker_weights/siamban_r50_l234_otb/config.yaml  --model_iter=4_net_G.pth --case=2 --eps=16  --trajcase=SE   --attack_universal --istargeted
