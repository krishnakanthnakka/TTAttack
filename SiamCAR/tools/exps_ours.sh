


python test_attack_ours.py  --dataset OTB100  --snapshot ../tracker_weights/model_general.pth   --model_iter=4_net_G.pth --case=54 --eps=8  --attack_universal


python test_attack_ours.py  --dataset UAV123  --snapshot snapshot/model_general.pth   --model_iter=8_net_G.pth --case=22 --eps=8
python test_attack_ours.py  --dataset UAV123  --snapshot snapshot/model_general.pth   --model_iter=8_net_G.pth --case=22 --eps=8 --attack_universal


python test_attack_ours.py  --dataset UAV123  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=54 --eps=8
python test_attack_ours.py  --dataset UAV123  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=54 --eps=8  --attack_universal

python test_attack_ours.py  --dataset UAV123  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=130 --eps=16
python test_attack_ours.py  --dataset UAV123  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=130 --eps=16 --attack_universal


python test_attack_ours.py  --dataset UAV123  --snapshot snapshot/model_general.pth   --model_iter=8_net_G.pth --case=138 --eps=16
python test_attack_ours.py  --dataset UAV123  --snapshot snapshot/model_general.pth   --model_iter=8_net_G.pth --case=138 --eps=16 --attack_universal


python test_attack_ours.py  --dataset UAV123  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=54 --eps=8  --attack_universal



python test_attack_ours.py  --dataset LaSOT  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=54 --eps=8  --attack_universal
python test_attack_ours.py  --dataset LaSOT  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=130 --eps=16 --attack_universal
python test_attack_ours.py  --dataset LaSOT  --snapshot snapshot/model_general.pth   --model_iter=8_net_G.pth --case=22 --eps=8 --attack_universal
python test_attack_ours.py  --dataset LaSOT  --snapshot snapshot/model_general.pth   --model_iter=8_net_G.pth --case=138 --eps=16 --attack_universal




python test_attack_ours.py  --dataset LaSOT  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=54 --eps=8
python test_attack_ours.py  --dataset LaSOT  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=130 --eps=16
python test_attack_ours.py  --dataset LaSOT  --snapshot snapshot/model_general.pth   --model_iter=8_net_G.pth --case=22 --eps=8
python test_attack_ours.py  --dataset LaSOT  --snapshot snapshot/model_general.pth   --model_iter=8_net_G.pth --case=138 --eps=16



python eval.py   --tracker_path ./results_OURS_54  --dataset UAV123  --tracker_prefix snapshot
python eval.py   --tracker_path ./results  --dataset OTB100  --tracker_prefix snapshot
python eval.py   --tracker_path ./results_U_OURS_54  --dataset UAV123  --tracker_prefix snapshot






python test_attack_ours.py  --dataset LaSOT  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=54 --eps=8



### HELPERS. Do not touch below

#---------------------------------------------- Running attacks with our attacks ----------------------------------------------
# image dependent
#  python test_attack_ours.py  --dataset OTB100  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=54 --eps=8
# universal
# python test_attack_ours.py  --dataset OTB100  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=54 --eps=8  --attack_universal

