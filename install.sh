#!/bin/bash
conda create -n adlcv python=3.10.10
conda activate adlcv
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
