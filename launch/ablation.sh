#!/bin/bash

CUDA_VISIBLE_DEVICES=1 nohup uv run expts/ablation_sgns.py > launch/logs/ablation_sgns.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup uv run expts/ablation_quartic.py > launch/logs/ablation_quartic.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup uv run expts/ablation_omn.py > launch/logs/ablation_omn.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup uv run expts/ablation_qwem.py > launch/logs/ablation_qwem.log 2>&1 &
