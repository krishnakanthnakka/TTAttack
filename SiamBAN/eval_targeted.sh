# ----- Ours#
cd experiments/siamban_r50_l234_otb
python ../../tools/eval_target.py --tracker_path ./results_Universal_Targeted_TTA_2 --dataset OTB100 --num 1  --tracker_prefix 'model' --trajcase=SE  #--s
