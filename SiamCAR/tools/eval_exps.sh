for ((case=250;case<=269;case++)); do
    python eval.py  --tracker_path ./results_U_OURS_$case  --dataset OTB100  --tracker_prefix snapshot
done
exit 1


python eval.py  --tracker_path ./results_U_OURS_185  --dataset OTB100  --tracker_prefix snapshot
python eval.py  --tracker_path ./results_U_OURS_186  --dataset OTB100  --tracker_prefix snapshot
python eval.py  --tracker_path ./results_U_OURS_187  --dataset OTB100  --tracker_prefix snapshot
python eval.py  --tracker_path ./results_U_OURS_188  --dataset OTB100  --tracker_prefix snapshot
# python eval.py  --tracker_path ./results_U_OURS_196  --dataset OTB100  --tracker_prefix snapshot
# python eval.py  --tracker_path ./results_U_OURS_161  --dataset OTB100  --tracker_prefix snapshot
# python eval.py  --tracker_path ./results_U_OURS_177  --dataset OTB100  --tracker_prefix snapshot
# python eval.py  --tracker_path ./results_U_OURS_22  --dataset OTB100  --tracker_prefix snapshot
