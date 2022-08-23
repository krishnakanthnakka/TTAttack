# for ((TRAJ=11;TRAJ<=14;TRAJ++)); do
#     python test_attack_ours_target.py  --dataset OTB100  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=211 --eps=16 --attack_universal --trajcase=$TRAJ  --targetcase=$TRAJ  --directions=4 #--vis
# done
# exit 1


python test_attack_ours_target.py  --dataset OTB100  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=$CASE --eps=16 --attack_universal --trajcase=14  --targetcase=14 #--vis



exit 1



for ((CASE=334;CASE<=339;CASE++)); do
    python test_attack_ours_target.py  --dataset OTB100  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=$CASE --eps=16 --attack_universal --trajcase=14  --targetcase=14 #--vis
done

exit 1

#python test_attack_ours.py  --dataset VOT2018  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=54 --eps=8
# python test_attack_ours_target.py  --dataset VOT2018  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=211 --eps=16 --attack_universal --trajcase=11  --targetcase=11 #--vis
# python test_attack_ours_target.py  --dataset VOT2018  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=211 --eps=16 --attack_universal --trajcase=12  --targetcase=12 #--vis
# python test_attack_ours_target.py  --dataset VOT2018  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=211 --eps=16 --attack_universal --trajcase=13  --targetcase=13 #--vis
# python test_attack_ours_target.py  --dataset VOT2018  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=211 --eps=16 --attack_universal --trajcase=14  --targetcase=14 #--vis
# python test_attack_ours_target.py  --dataset VOT2018  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=211 --eps=16 --attack_universal --trajcase=21  --targetcase=21 #--vis
# python test_attack_ours_target.py  --dataset VOT2018  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=211 --eps=16 --attack_universal --trajcase=22  --targetcase=22 #--vis
# python test_attack_ours_target.py  --dataset VOT2018  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=211 --eps=16 --attack_universal --trajcase=23  --targetcase=23 #--vis
# python test_attack_ours_target.py  --dataset VOT2018  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=211 --eps=16 --attack_universal --trajcase=24  --targetcase=24 #--vis


# python test_attack_ours_target.py  --dataset UAV123  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=211 --eps=16 --attack_universal --trajcase=11  --targetcase=11 #--vis
# python test_attack_ours_target.py  --dataset UAV123  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=211 --eps=16 --attack_universal --trajcase=12  --targetcase=12 #--vis
# python test_attack_ours_target.py  --dataset UAV123  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=211 --eps=16 --attack_universal --trajcase=13  --targetcase=13 #--vis
# python test_attack_ours_target.py  --dataset UAV123  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=211 --eps=16 --attack_universal --trajcase=14  --targetcase=14 #--vis
# python test_attack_ours_target.py  --dataset UAV123  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=211 --eps=16 --attack_universal --trajcase=21  --targetcase=21 #--vis
# python test_attack_ours_target.py  --dataset UAV123  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=211 --eps=16 --attack_universal --trajcase=22  --targetcase=22 #--vis
# python test_attack_ours_target.py  --dataset UAV123  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=211 --eps=16 --attack_universal --trajcase=23  --targetcase=23 #--vis
# python test_attack_ours_target.py  --dataset UAV123  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=211 --eps=16 --attack_universal --trajcase=24  --targetcase=24 #--vis




python test_attack_ours.py  --dataset OTB100  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=185 --eps=8 --attack_universal
python test_attack_ours.py  --dataset OTB100  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=186 --eps=8 --attack_universal
python test_attack_ours.py  --dataset OTB100  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=187 --eps=8 --attack_universal
python test_attack_ours.py  --dataset OTB100  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=188 --eps=8 --attack_universal

exit 1




python test_attack_ours.py  --dataset OTB100  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=190 --eps=8 --attack_universal
python test_attack_ours.py  --dataset OTB100  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=191 --eps=8 --attack_universal
python test_attack_ours.py  --dataset OTB100  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=192 --eps=8 --attack_universal
python test_attack_ours.py  --dataset OTB100  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=193 --eps=8 --attack_universal
python test_attack_ours.py  --dataset OTB100  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=196 --eps=8 --attack_universal
python test_attack_ours.py  --dataset OTB100  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=161 --eps=8 --attack_universal
python test_attack_ours.py  --dataset OTB100  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=177 --eps=8 --attack_universal
#python test_attack_ours.py  --dataset OTB100  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=191 --eps=8 --attack_universal






# python test_attack_ours.py  --dataset VOT2018  --snapshot snapshot/model_general.pth   --model_iter=8_net_G.pth --case=22 --eps=8
# python test_attack_ours.py  --dataset VOT2018  --snapshot snapshot/model_general.pth   --model_iter=8_net_G.pth --case=22 --eps=8 --attack_universal
# python test_attack_ours.py  --dataset VOT2018  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=130 --eps=8
# python test_attack_ours.py  --dataset VOT2018  --snapshot snapshot/model_general.pth   --model_iter=4_net_G.pth --case=130 --eps=8 --attack_universal
# python test_attack_ours.py  --dataset VOT2018  --snapshot snapshot/model_general.pth   --model_iter=8_net_G.pth --case=138 --eps=8
# python test_attack_ours.py  --dataset VOT2018  --snapshot snapshot/model_general.pth   --model_iter=8_net_G.pth --case=138 --eps=8 --attack_universal
