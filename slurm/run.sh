#!/bin/bash
#SBATCH --job-name=ocean_ml
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=13:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu

module purge
module load WebProxy

export http_proxy=http://10.73.132.63:8080
export https_proxy=http://10.73.132.63:8080

PROJECT_DIR="/scratch/user/aaupadhy/college/RA/FlowNet"
cd $PROJECT_DIR
echo "Working directory: $PWD"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
export CARTOPY_OFFLINE_SHARED="$PROJECT_DIR/outputs/cartopy_shapefiles"
echo "PYTHONPATH: $PYTHONPATH"


export DASK_DASHBOARD_PORT=8787
COMPUTE_NODE=$(hostname -s)  # Get short hostname (e.g., g077)

echo "To access Dask dashboard, run this command on your local machine:"
echo "ssh -N -L 8787:${COMPUTE_NODE}.grace.hprc.tamu.edu:8787 aaupadhy@grace.hprc.tamu.edu"
echo "Then visit http://localhost:8787 in your browser"

source ~/.bashrc
conda activate ML

export CUDA_VISIBLE_DEVICES=0
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export OMP_NUM_THREADS=8

echo "Job started at $(date)"
echo "Running on ${COMPUTE_NODE}.grace.hprc.tamu.edu"
echo "GPU information:"
nvidia-smi

python $PROJECT_DIR/main.py --mode all

# Print job completion info
echo "Job finished at $(date)"
