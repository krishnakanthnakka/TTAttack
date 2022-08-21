
# torch - 1.5.0, torchvision- 0.6.0

git config --global user.email krishna.nakka@epfl.ch
git config --global user.name krishnakanthnakka
git config --global credential.helper 'cache --timeout 72000'


DIR=/cvlabsrc1/home/krishna/TTA/TTAttack/SiamRPNpp


# add them for ocean tracker
export PYTHONPATH=$DIR/pysot/pysot/tracker/:$PYTHONPATH
export PYTHONPATH=$DIR/pysot/pysot/tracker/ocean_utils:$PYTHONPATH

export PYTHONPATH=$DIR:$PYTHONPATH
export PYTHONPATH=$DIR/pysot/:$PYTHONPATH
export PYTHONPATH=$DIR/pix2pix/:$PYTHONPATH
pip install dominate yacs colorama icecream --user


# IMPORTANT
# torchvision.__version__: 0.6.0,  torch: 1.5.0

#pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

pip install torch==1.5.0 torchvision==0.6.0
export PYTHONDONTWRITEBYTECODE=1

# torch 1.2.0 works
