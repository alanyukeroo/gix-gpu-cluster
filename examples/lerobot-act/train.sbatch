#!/bin/bash
# LeRobot example: train an ACT policy on a dataset you recorded, inside the official LeRobot container.
#
#   lerobot-act/
#   ├── train.sbatch          ← this file
#   └── data/my-dataset/      ← your LeRobot dataset (meta/, data/, videos/), e.g. from lerobot-record
#
# From your laptop:   gix upload lerobot-act   then   gix submit lerobot-act/train.sbatch

#SBATCH --job-name=lerobot-act      # CHANGE: name shown in `gix status`
#SBATCH --gres=gpu:1                # whole RTX 5090. shard:1 is enough for a quick test
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00             # CHANGE: max run time (hh:mm:ss)

# CHANGE: training settings
DATASET=data/my-dataset             # folder of your dataset, inside this project folder
POLICY=act                          # act, diffusion, pi0, smolvla, ...
STEPS=50000                         # training steps
BATCH=8

# Official LeRobot image: PyTorch with CUDA 12.8 (works on the RTX 5090), LeRobot, ffmpeg
# Pinned: :latest changes without notice and has removed flags this script uses
IMAGE=docker.io/huggingface/lerobot-gpu@sha256:9760904226c7c1c109d3756d356f5987892a279da90477c8426c0581186c893d

set -o pipefail
if [ ! -f "$DATASET/meta/info.json" ]; then
    echo "ERROR: no LeRobot dataset at $DATASET (expected $DATASET/meta/info.json)."
    echo "Folders with a dataset in this project:"
    find . -path ./outputs -prune -o -name info.json -path '*/meta/*' -print | sed 's|^\./||; s|/meta/info.json$||; s|^|  |'
    exit 1
fi
NAME=${POLICY}-${SLURM_JOB_ID}
# LeRobot writes checkpoints with symlinks, which the NAS doesn't support. So training writes to the
# node's local disk, and outputs/ on the NAS gets a copy every 10 minutes and when the job ends.
SCRATCH=/var/lib/gix-containers/$USER/scratch/$SLURM_JOB_ID
mkdir -p "$SCRATCH" outputs .cache
sync_outputs() { [ -d "$SCRATCH/run" ] && rsync -rq --no-links "$SCRATCH/run/" "outputs/$NAME/"; }
( while sleep 600; do sync_outputs; done ) &
SYNCER=$!
trap 'kill $SYNCER 2>/dev/null; sync_outputs; podman unshare rm -rf "$SCRATCH"' EXIT
echo "job $SLURM_JOB_ID on $(hostname): $POLICY on $DATASET, $STEPS steps -> outputs/$NAME"

# Your project folder is /work inside the container. The first run on a node downloads the image (~6 GB).
podman run --rm \
    --device=nvidia.com/gpu=all \
    --userns=keep-id --user "$(id -u):$(id -g)" \
    --shm-size=8g \
    -e HOME=/tmp \
    -e TORCH_HOME=/work/.cache/torch \
    -e HF_HOME=/work/.cache/huggingface \
    -v "$PWD":/work -w /work \
    -v "$SCRATCH":/scratch \
    "$IMAGE" \
    lerobot-train \
        --dataset.repo_id=local/$(basename "$DATASET") \
        --dataset.root="$DATASET" \
        --policy.type="$POLICY" \
        --policy.device=cuda \
        --policy.push_to_hub=false \
        --trackio.enable=false \
        --output_dir=/scratch/run \
        --batch_size="$BATCH" \
        --steps="$STEPS" \
        --num_workers="$SLURM_CPUS_PER_TASK" \
    2>&1 | grep --line-buffered -v 'level=warning'
