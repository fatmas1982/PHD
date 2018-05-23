#! /bin/bash
#BATCH --job-name=Cluster_Graph_Process_Topic
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
###SBATCH --ntasks=1
#SBATCH --array=0-2
###SBATCH --cpus-per-task=1
####SBATCH --exclusive


#SBATCH --output=Cluster_Graph_Process_Topic_%A.stdout
#SBATCH --out=Cluster_Graph_Process_Topic_%A.txt

#SBATCH --error=Cluster_Graph_Process_Topic_%A.stderr
#SBATCH --export=ALL
#SBATCH --exclude=comp004,comp005

module load CUDA/8.0.44
module load cuDNN/5.1-CUDA-8.0.44


module load Singularity/2.2.1

unset JAVA_HOME

export TEST_TMPDIR=~/data/bazel-cache

export CUDA_VISIBLE_DEVICES=${SLURM_ARRAY_TASK_ID}

all_gpus=1

cuda=${SLURM_ARRAY_TASK_ID}

processor='gpu'
dataset='split'

singularity exec ~/data/containers/TensorFlow/tf-gpu.img ~/gpu/split/CJ-3/Cluster_Graph_Process_Topic.py ${SLURM_ARRAY_TASK_ID} $all_gpus $cuda $processor $dataset






