#!/bin/sh
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J lef_test4
#BSUB -n 4
#BSUB -W 1:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
echo "Running script..."
echo $PWD
nvidia-smi
# module swap python3/3.9.11
# module swap cuda/11.6

source ../adlcv_venv/bin/activate
python aisynthesizer/train_model.py