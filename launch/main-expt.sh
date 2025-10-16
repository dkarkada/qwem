#!/bin/bash

CUDA_VISIBLE_DEVICES=1 nohup uv run expts/qwem_small.py > launch/logs/qs.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup uv run expts/qwem_large.py > launch/logs/ql.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup uv run expts/sgns_small.py > launch/logs/sgs.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup uv run expts/sgns_large.py > launch/logs/sgl.log 2>&1 &
