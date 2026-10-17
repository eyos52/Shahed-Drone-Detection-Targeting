#!/bin/bash
#SBATCH --job-name=real_eval
#SBATCH --account=klynes_cs5351_1267_801c_0001
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/users/apgoel/senior_design/runs/real_eval.%j.log

# Usage: sbatch real_eval.sbatch [extra real_eval.py args, e.g. --source /path/to/video.mp4]
set -e
cd /users/apgoel/senior_design/ti_team_repo/scripts
~/.conda/envs/drone_ti/bin/python -u real_eval.py "$@"
