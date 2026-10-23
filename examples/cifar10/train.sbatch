#!/bin/bash
# CIFAR-10 example: train a small image classifier on the GIX GPU cluster.
# From your laptop:   gix upload cifar10   then   gix submit cifar10/train.sbatch

#SBATCH --job-name=cifar10
#SBATCH --gres=shard:1              # a quarter of an RTX 5090 is plenty for this
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00

IMAGE=docker.io/pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
RUN='pip install -q -r requirements.txt && python train.py --data data --out outputs --epochs 30'

set -o pipefail
echo "job $SLURM_JOB_ID on $(hostname)"
podman run --rm \
    --device=nvidia.com/gpu=all \
    --userns=keep-id \
    --shm-size=8g \
    -e HOME=/tmp \
    -v "$PWD":/work -w /work \
    "$IMAGE" bash -c "$RUN" 2>&1 | grep --line-buffered -v 'level=warning'
