cp -r experiments/  TTA/
cp -r pysotcar/  TTA/
cp -r toolkit/  TTA/
cp -r tools/  TTA/
rm -r TTA/tools/snapshot/model_general.pth


cp   ./*.py  TTA/
cp  ./*.jpg  TTA/
cp  ./*.png  TTA/
cp  ./*.sh  TTA/
cp  ./*.xlsx  TTA/
cp   ./*.md  TTA/



cd TTA
git add ./
git commit -m $1
git push origin SiamCAR
cd ..
