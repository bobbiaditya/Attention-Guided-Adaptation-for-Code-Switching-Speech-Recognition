#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16          # <- match to OMP_NUM_THREADS
#SBATCH --mem=60000M
#SBATCH --partition=gpuA40x4-interactive        # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bbjs-delta-gpu
#SBATCH --job-name=profile
#SBATCH --time=1:00:00           # hh:mm:ss for the job

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./path.sh

set -x

second_all="10 15 20"                               # audio length in seconds
fs=16000                                            # sampling rate
asr_model_file=/home/espnet/egs2/seame/asr1/exp/asr_seame_whisper_small_zh_en_adapter_2block_192/valid.acc.ave.pth    # assuming config.yaml exists in same directory

for second in ${second_all}; do
    python pyscripts/utils/profile_encoder.py \
        --second ${second} \
        --fs ${fs} \
        --model_file "${asr_model_file}"
done
