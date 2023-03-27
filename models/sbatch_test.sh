#!/bin/bash
#SBATCH --constraint='a100'
source /itet-stor/calmagro/net_scratch/conda/etc/profile.d/conda.sh
conda activate big
python -u train.py  "$@"





