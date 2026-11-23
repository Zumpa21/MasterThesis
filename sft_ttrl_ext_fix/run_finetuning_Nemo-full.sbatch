#!/bin/bash
#SBATCH --job-name=sft_ttrl_ext_Nemo-full
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your.email@example.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --partition=l40
#SBATCH --output=/path/to/MasterThesis/sft_ttrl_ext/finetune_output_0_25.log
#SBATCH --chdir=/path/to/MasterThesis/sft_ttrl_ext
#SBATCH --gres=gpu:1

export HF_HOME="/path/to/cache/huggingface"
export PYTHONPATH="/path/to/MasterThesis/arc-prize-2024/"
export TRITON_CACHE_DIR="/path/to/cache/triton"
source /path/to/venv/bin/activate

/path/to/venv/bin/python3 run_finetuning_Nemo-full.py
