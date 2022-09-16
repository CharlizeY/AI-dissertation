#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=yy3219 # required to send email notifcations - please replace <your_username> with your college login name or email address

# venv setup
export PATH=/vol/bitbucket/${USER}/cbm_venv/bin/:$PATH
source activate

# setup working environment
export PYTHONPATH=${PYTHONPATH}:/vol/bitbucket/${USER}/roko-for-charlize
source /vol/cuda/11.2.1-cudnn8.1.0.77/setup.sh
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi

# Clean up cache to avoid tampering with experiments
rm -f labels_100.pkl data.npy

# Run the actual model
cd /vol/bitbucket/${USER}/roko-for-charlize
for i in {1..5}
do
  # python img_classifier.py --fold ${i}
  # python img_classifier_upsample.py --fold ${i}
  python img_classifier_downsample.py --fold ${i}
  # python transformer_experiment.py --fold ${i}
done

# How long did the script run for
uptime

