#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# train grid
python main.py --batch_size 256 \
               --model  soft \
               --feature_dir bottom_up_origin \
               --cpu_size 128 \
               --output saved_models/bottom_up_origin/first_try
