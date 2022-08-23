python eval_target.py  --tracker_path ./results_U_OURS_211_23  --dataset OTB100  --tracker_prefix snapshot --trajcase=11
python eval_target.py  --tracker_path ./results_U_OURS_211_23  --dataset OTB100  --tracker_prefix snapshot --trajcase=12
python eval_target.py  --tracker_path ./results_U_OURS_211_23 --dataset OTB100  --tracker_prefix snapshot --trajcase=13
python eval_target.py  --tracker_path ./results_U_OURS_211_23  --dataset OTB100  --tracker_prefix snapshot --trajcase=14



python eval_target.py  --tracker_path ./results_Universal_Targeted_TTA_2  --dataset OTB100  --tracker_prefix model_general --trajcase=11


exit 1



for ((case=333;case<=339;case++)); do
# python eval_target.py  --tracker_path ./results_U_OURS_$case  --dataset OTB100  --tracker_prefix snapshot --trajcase=11
# python eval_target.py  --tracker_path ./results_U_OURS_$case  --dataset OTB100  --tracker_prefix snapshot --trajcase=12
# python eval_target.py  --tracker_path ./results_U_OURS_$case  --dataset OTB100  --tracker_prefix snapshot --trajcase=13
python eval_target.py  --tracker_path ./results_U_OURS_{$case}_12  --dataset OTB100  --tracker_prefix snapshot --trajcase=14
done

exit 1

python eval_target.py  --tracker_path ./results_U_OURS_211  --dataset LaSOT  --tracker_prefix snapshot --trajcase=11
python eval_target.py  --tracker_path ./results_U_OURS_211  --dataset LaSOT  --tracker_prefix snapshot --trajcase=12
python eval_target.py  --tracker_path ./results_U_OURS_211  --dataset LaSOT  --tracker_prefix snapshot --trajcase=13
python eval_target.py  --tracker_path ./results_U_OURS_211  --dataset LaSOT  --tracker_prefix snapshot --trajcase=14
python eval_target.py  --tracker_path ./results_U_OURS_211  --dataset LaSOT  --tracker_prefix snapshot --trajcase=21
python eval_target.py  --tracker_path ./results_U_OURS_211  --dataset LaSOT  --tracker_prefix snapshot --trajcase=22
python eval_target.py  --tracker_path ./results_U_OURS_211  --dataset LaSOT  --tracker_prefix snapshot --trajcase=23
python eval_target.py  --tracker_path ./results_U_OURS_211  --dataset LaSOT  --tracker_prefix snapshot --trajcase=24


python eval_target.py  --tracker_path ./results_U_OURS_338_12  --dataset OTB100  --tracker_prefix snapshot --trajcase=14
python eval_target.py  --tracker_path ./results_U_OURS_339_12  --dataset OTB100  --tracker_prefix snapshot --trajcase=14
