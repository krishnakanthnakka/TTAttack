

cd experiments/siamban_r50_l234_otb
python -u ../../tools/test_attack_ours.py --snapshot tracker_weights/model.pth  --dataset OTB100 --config tracker_weights/config.yaml  --model_iter=4_net_G.pth --case=54 --eps=8  --attack_universal
