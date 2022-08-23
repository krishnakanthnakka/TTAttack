#DIR=/cvlabsrc1/home/krishna/TTA/TTAttack/SiamRPNpp
DIR=`pwd`
export PYTHONPATH=$DIR/pysot/pysot/tracker/:$PYTHONPATH
export PYTHONPATH=$DIR/pysot/pysot/tracker/ocean_utils:$PYTHONPATH
export PYTHONPATH=$DIR:$PYTHONPATH
export PYTHONPATH=$DIR/pysot/:$PYTHONPATH
export PYTHONPATH=$DIR/pix2pix/:$PYTHONPATH






pip install dominate yacs colorama icecream --user
pip install torch==1.5.0 torchvision==0.6.0
export PYTHONDONTWRITEBYTECODE=1

