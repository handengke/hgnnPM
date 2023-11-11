#!/bin/bash

dataset=${1:-imdb}

# the naive pm without any optimizations
python ./pm_naive.py --dataset=$dataset 
# the pipeline pm
python ./pm_pipeline.py --dataset=$dataset
# the fine-grained stage fusion pm
python ./pm_gpu.py --dataset=$dataset
