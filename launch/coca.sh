#! /bin/bash

CUDA_VISIBLE_DEVICES=1 nohup uv run expts/sgns_cocanow.py > launch/logs/coca.log 2>&1 &