# update conda
conda update conda

# create separate environment
conda create -n DeepLungQuant python=3.6 anaconda

# enter the environment
conda activate DeepLungQuant

# install antspy for affine and SyN registration
python3 -m pip install antspyx-0.2.2-cp36-cp36m-linux_x86_64.whl

# install necessary packages listed in requirements.txt
python3 -m pip install -r requirements.txt
