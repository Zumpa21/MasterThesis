#!/bin/bash
#SBATCH --job-name=merge_orig_0_25_finetuned_only_eval_Nemo-full
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your.email@example.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --partition=rtx2080
#SBATCH --output=/path/to/MasterThesis/orig_0_25_dataset_finetuned_only/merge_eval_output_0_25.log
#SBATCH --chdir=/path/to/MasterThesis/orig_0_25_dataset_finetuned_only
#SBATCH --gres=gpu:1

export HF_HOME="/path/to/cache/huggingface"
export PYTHONPATH="/path/to/MasterThesis/orig_0_25_dataset_finetuned_only/"
export TRITON_CACHE_DIR="/path/to/cache/triton"

# Fix PyTorch CUDA VRAM fragmentation
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Start background VRAM monitoring (Logs VRAM every 5 seconds to vram_log.txt)
nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu --format=csv -l 5 > vram_log.txt &
MONITOR_PID=$!

source /path/to/venv/bin/activate
PY=/path/to/venv/bin/python3

# Sequential Shard Loop
N_SHARDS=4

# Merge all shards into final output
$PY merge_submissions.py $N_SHARDS

# Stop the background monitor process
kill $MONITOR_PID
