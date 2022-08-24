

# for OTB100
# cd experiments/siamban_r50_l234_otb
# python -u ../../tools/test_attack_ours.py --snapshot ../../../tracker_weights/siamban_r50_l234_otb/model.pth  --dataset OTB100 --config ../../../tracker_weights/siamban_r50_l234_otb/config.yaml  --model_iter=4_net_G.pth --case=1 --eps=8  --attack_universal




# for VOT2018
cd experiments/siamban_r50_l234
python -u ../../tools/test_attack_ours.py --snapshot ../../../tracker_weights/siamban_r50_l234/model.pth  --dataset VOT2018 --config ../../../tracker_weights/siamban_r50_l234/config.yaml  --model_iter=4_net_G.pth --case=1 --eps=8  --attack_universal #--video=ball1 --vis

