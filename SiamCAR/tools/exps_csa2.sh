

python test_attack_csaTS.py  --dataset OTB100  --snapshot snapshot/model_general.pth   --model_iter=8_net_G.pth --case=3 --eps=8
python test_attack_csaTS.py  --dataset VOT2018  --snapshot snapshot/model_general.pth   --model_iter=8_net_G.pth --case=3 --eps=8
python test_attack_csaTS.py  --dataset UAV123  --snapshot snapshot/model_general.pth   --model_iter=8_net_G.pth --case=3 --eps=8





# python eval.py   --tracker_path ./results_OURS_54  --dataset UAV123  --tracker_prefix snapshot
# python eval.py   --tracker_path ./results  --dataset OTB100  --tracker_prefix snapshot
# python eval.py   --tracker_path ./results_U_OURS_54  --dataset UAV123  --tracker_prefix snapshot




### HELPERS. Do not touch below

#---------------------------------------------- Running attacks with our attacks ----------------------------------------------
# image dependent
#  python test_attack_ours.py  --dataset OTB100  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=54 --eps=8
# universal
# python test_attack_ours.py  --dataset OTB100  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=54 --eps=8  --attack_universal

