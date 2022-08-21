python train1.py --eps=8 --case=210  --tracker_name=siamrpn_r50_l234_dwxcorr --istargeted
python train1.py --eps=16 --case=211  --tracker_name=siamrpn_r50_l234_dwxcorr --istargeted


python train1.py --eps=8 --case=212  --tracker_name=siamrpn_r50_l234_dwxcorr_lt --istargeted
python train1.py --eps=16 --case=213  --tracker_name=siamrpn_r50_l234_dwxcorr_lt --istargeted

python train1.py --eps=16 --case=214  --tracker_name=siamrpn_r50_l234_dwxcorr --istargeted

python train1.py --eps=16 --case=216  --tracker_name=siamrpn_r50_l234_dwxcorr --istargeted
python train1.py --eps=16 --case=217  --tracker_name=siamrpn_r50_l234_dwxcorr_lt --istargeted


python train1.py --eps=16 --case=218  --tracker_name=siamrpn_r50_l234_dwxcorr --istargeted
python train1.py --eps=16 --case=219  --tracker_name=siamrpn_r50_l234_dwxcorr --istargeted


python train1.py --eps=16 --case=219  --tracker_name=siamrpn_r50_l234_dwxcorr_lt --istargeted


python train1.py --eps=16 --case=221  --tracker_name=siamrpn_r50_l234_dwxcorr_lt --istargeted



python train1.py --eps=16 --case=189  --tracker_name=siamrpn_r50_l234_dwxcorr_lt --istargeted


python eval.py --tracker_path=./results --dataset=VOT2018-LT --model_epoch=6_net_G.pth --case=139  --tracker_prefix=G_template_L2_500_regress_siamrpn_r50_l234_dwxcorr_lt


python train1.py --eps=16 --case=189  --tracker_name=siamrpn_r50_l234_dwxcorr_lt


python train1.py --eps=16 --case=300  --tracker_name=siamrpn_alex_dwxcorr --istargeted
