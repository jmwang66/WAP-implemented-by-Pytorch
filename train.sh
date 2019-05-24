#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
nohup python -u ./train_wap.py  1>log.txt 2>&1 &
tail -f log.txt

