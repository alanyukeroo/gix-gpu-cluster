#!/bin/bash
# Template for a training job on the GIX GPU cluster.
# Copy it into your project folder, change the lines marked CHANGE, then from your laptop:
#   gix upload my-project
#   gix submit my-project/train.sbatch
#
# The job runs in the folder this file is in, so relative paths (data/, outputs/) work.

#SBATCH --job-name=train            # CHANGE: a short name you will recognize in `gix status`
#SBATCH --gres=gpu:1                # whole RTX 5090 (32 GB). For small jobs: shard:1 (shares the GPU)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00             # CHANGE: max run time (hh:mm:ss). The job stops after this

# CHANGE: the container image with your framework (CUDA >= 12.8 for the RTX 5090)
IMAGE=docker.io/pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

# CHANGE: what to run inside the container
RUN='pip install -q -r requirements.txt && python train.py --data data/ --out outputs/'

echo "job $SLURM_JOB_ID on $(hostname)"
mkdir -p outputs

# Your project folder is /work inside the container. The first run on a node downloads
# the image (a few minutes); later runs start right away.
podman run --rm \
    --device=nvidia.com/gpu=all \
    --userns=keep-id \
    --shm-size=8g \
    -e HOME=/tmp \
    -v "$PWD":/work -w /work \
    "$IMAGE" bash -c "$RUN"
