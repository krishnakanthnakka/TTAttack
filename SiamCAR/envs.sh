

DIR=`pwd`

export PYTHONPATH=$DIR/../SiamRPNpp/pysot/pysot/tracker/:$PYTHONPATH
export PYTHONPATH=$DIR/../SiamRPNpp/pysot/pysot/tracker/ocean_utils:$PYTHONPATH
export PYTHONPATH=$DIR/../SiamRPNpp/:$PYTHONPATH
export PYTHONPATH=$DIR/../SiamRPNpp/pysot/:$PYTHONPATH
export PYTHONPATH=$DIR/../SiamRPNpp/pix2pix/:$PYTHONPATH
pip install dominate yacs colorama icecream --user
pip install torch==1.5.0 torchvision==0.6.0
export PYTHONDONTWRITEBYTECODE=1


DIR=`pwd`
export PYTHONPATH=$DIR:$PYTHONPATH
