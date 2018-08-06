#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# bottom_up_origin
# python bottom_up_origin.py --cuda

# bottom_up_stage4
python bottom_up_stage4.py --cuda
