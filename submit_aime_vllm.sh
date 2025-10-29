#!/bin/bash
#SBATCH --job-name=aime-vllm
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Exit on error, undefined variable usage, or failed pipe segments.
set -euo pipefail

# Uncomment and customize the following lines to load your runtime environment.
# module load anaconda
# source activate your-env-name
# or: source /path/to/venv/bin/activate

# Ensure the log directory exists before writing output.
mkdir -p logs

# Change into the project directory if the job starts elsewhere.
cd "$SLURM_SUBMIT_DIR"

# Run the evaluation script. Override the defaults by passing arguments to sbatch,
# e.g. `sbatch submit_aime_vllm.sh --questions 0 1 2`.
python aime_vllm_deepseek.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --questions 0 13 \
  --attempts 7 \
  --logprobs 5 \
  --max-new-tokens 512 \
  --temperature 0.8 \
  --top-p 0.95 \
  --out generations.parquet \
  "$@"
