# ----- OTb100
# cd experiments/siamban_r50_l234_otb
# python ../../tools/eval.py --tracker_path ./results_Universal_TTA_1 --dataset OTB100 --num 1  --tracker_prefix 'model'



# ----- VOT2018
cd experiments/siamban_r50_l234
python ../../tools/eval.py --tracker_path ./results_Universal_TTA_1 --dataset VOT2018 --num 1  --tracker_prefix 'model'
