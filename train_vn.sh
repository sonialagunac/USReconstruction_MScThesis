#!/bin/bash
#SBATCH --output=out_log/%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --job-name=ESRGAN
#SBATCH --mail-type=END,FAIL

export DATA_PATH="/scratch_net/biwidl307/sonia/data_original/"
export EXP_PATH="/scratch_net/biwidl307/sonia/USImageReconstruction-Sonia/runs/"
export PYTHONPATH=": /scratch_net/biwidl307/sonia/USImageReconstruction-Sonia/"

source /itet-stor/slaguna/net_scratch/conda/etc/profile.d/conda.sh
conda activate master_thesis
cd /scratch_net/biwidl307/sonia/USImageReconstruction-Sonia
python -u codes/VNSolver/train_vn_sos.py "$@"


