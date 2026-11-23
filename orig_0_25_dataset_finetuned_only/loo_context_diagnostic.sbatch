#!/bin/bash
#SBATCH --job-name=loo_context_diag_orig_0_25
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your.email@example.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=54G
#SBATCH --partition=l40
#SBATCH --output=/path/to/MasterThesis/orig_0_25_dataset_finetuned_only/loo_context_diag_unaugmented.log
#SBATCH --chdir=/path/to/MasterThesis/orig_0_25_dataset_finetuned_only
#SBATCH --gres=gpu:1

export HF_HOME="/path/to/cache/huggingface"
export PYTHONPATH="/path/to/MasterThesis/orig_0_25_dataset_finetuned_only/"
export TRITON_CACHE_DIR="/path/to/cache/triton"

# Fix PyTorch CUDA VRAM fragmentation
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Start background VRAM monitoring (Logs VRAM every 5 seconds to vram_log.txt)
nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu --format=csv -l 5 > vram_log_loo_diag.txt &
MONITOR_PID=$!

source /path/to/venv/bin/activate
PY=/path/to/venv/bin/python3

MODEL_PATH="/path/to/MasterThesis/orig_0_25_dataset_finetuned_only/pretrained_models/Mistral-NeMo-Minitron-Full-merged"

# Single pass, no sharding needed -- this is a fast, non-training diagnostic.
# Kept the retry pattern from the eval sbatch in case of a transient node/GPU issue.
MAX_ATTEMPTS=3
for attempt in $(seq 1 $MAX_ATTEMPTS); do
    echo "=== LOO context diagnostic, Attempt $attempt/$MAX_ATTEMPTS ==="
    $PY loo_context_diagnostic.py "$MODEL_PATH"
    status=$?
    if [ $status -eq 0 ]; then
        echo "=== Diagnostic completed successfully ==="
        break
    fi
    sleep 10
done

# Stop the background monitor process
kill $MONITOR_PID
