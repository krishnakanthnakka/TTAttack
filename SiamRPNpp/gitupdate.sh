echo $1

cp -r utils/  TTA/
cp -r utils_TTA/  TTA/
cp -r jsons/  TTA/
cp -r pix2pix/ TTA/


cp -r pysot/build TTA/pysot/
cp -r pysot/demo TTA/pysot/
cp -r pysot/toolkit/ TTA/pysot/
cp -r pysot/pysot/ TTA/pysot/
cp -r pysot/vot_iter TTA/pysot/
cp -r pysot/training_dataset TTA/pysot/
cp -r pysot/testing_dataset TTA/pysot/


cp -r pysot/tools/*.py TTA/pysot/tools/
cp -r pysot/tools/*.sh TTA/pysot/tools/
cp -r pysot/tools/logs TTA/pysot/tools/
cp -r pysot/tools/results TTA/pysot/tools/
cp -r pysot/tools/results_U TTA/pysot/tools/
cp -r pysot/tools/results_clean/ TTA/pysot/tools/
cp -r pysot/tools/results_target/ TTA/pysot/tools/

cp   ./*.py  TTA/
cp  ./*.jpg  TTA/
cp  ./*.png  TTA/
cp  ./*.sh  TTA/
cp  ./*.xlsx  TTA/

git config --global credential.helper 'cache --timeout 72000'


cd TTA

# git lfs track "*pth"
# git add .gitattributes

git add ./
git commit -m $1
git push origin
cd ..
