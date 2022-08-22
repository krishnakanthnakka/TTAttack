# ----- Ours#
cd experiments/siamban_r50_l234_otb
python ../../tools/eval.py --tracker_path ./results_Universal_TTA_54 --dataset OTB100 --num 1  --tracker_prefix 'model'
