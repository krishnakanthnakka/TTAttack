cd experiments/siamban_r50_l234_otb
# for one traj
python -u ../../tools/test_attack_ours_targeted.py --snapshot ../../../tracker_weights/siamban_r50_l234_otb/model.pth  --dataset OTB100 --config ../../../tracker_weights/siamban_r50_l234_otb/config.yaml  --model_iter=4_net_G.pth --case=2 --eps=16  --trajcase=11  --targetcase=11  --attack_universal --istargeted
# python -u ../../tools/test_attack_ours_targeted.py --snapshot ../../../tracker_weights/siamban_r50_l234_otb/model.pth  --dataset OTB100 --config ../../../tracker_weights/siamban_r50_l234_otb/config.yaml  --model_iter=4_net_G.pth --case=2 --eps=16  --trajcase=11  --targetcase=11  --attack_universal --istargeted  --directions=12 --vis --video=Biker



# for ((TRAJ=11;TRAJ<=11;TRAJ++)); do
#     python -u ../../tools/test_attack_ours_targeted.py --snapshot ../../../tracker_weights/siamban_r50_l234_otb/model.pth  --dataset OTB100 --config ../../../tracker_weights/siamban_r50_l234_otb/config.yaml  --model_iter=4_net_G.pth --case=2 --eps=16  --trajcase=$TRAJ  --targetcase=$TRAJ  --attack_universal --istargeted  --directions=12 --vis
# done


# exit 1
