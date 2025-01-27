#!/bin/bash
#SBATCH --job-name=ocean_train
#SBATCH --output=%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=140G
#SBATCH --time=15:00:00

module purge
module load WebProxy
export http_proxy=http://10.73.132.63:8080
export https_proxy=http://10.73.132.63:8080

PROJECT_DIR="/scratch/user/aaupadhy/college/RA/FlowNet"
cd $PROJECT_DIR
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
export CARTOPY_OFFLINE_SHARED="$PROJECT_DIR/outputs/cartopy_shapefiles"

COMPUTE_NODE=$(hostname -s)
echo "ssh -N -L 8787:${COMPUTE_NODE}:8787 aaupadhy@grace.hprc.tamu.edu"

source ~/.bashrc
conda activate ML

export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export OMP_NUM_THREADS=8

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0

export CUDA_VISIBLE_DEVICES=0,1,2,3
export WORLD_SIZE=4
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export OMP_NUM_THREADS=8

MASTER_PORT=$(shuf -i 29500-29999 -n 1)

echo "Job started at $(date)"
echo "Running on ${COMPUTE_NODE}.grace.hprc.tamu.edu"
nvidia-smi

torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --master_port=$MASTER_PORT \
    --master_addr="127.0.0.1" \
    --node_rank=0 \
    --max_restarts=0 \
    --start_method=spawn \
    main.py --mode all

echo "Job finished at $(date)"