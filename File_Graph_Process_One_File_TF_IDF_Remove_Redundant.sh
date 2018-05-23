#! /bin/bash
#BATCH --job-name=first-cuda-job
#SBATCH --partition=cpu
###SBATCH --gres=gpu:1
###SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1-3
###SBATCH --exclusive
#SBATCH --exclude=comp0[01-19]
###SBATCH --nodelist=comp058 ###,comp070,comp069

#SBATCH --output=FJ1_%A.stdout

#SBATCH --out=FJ1_%A.txt

#SBATCH --error=FJ1_%A.stderr
#SBATCH --export=ALL


#module load CUDA/8.0.44
#module load cuDNN/5.1-CUDA-8.0.44


module load Singularity/2.2.1

unset JAVA_HOME

export TEST_TMPDIR=~/data/bazel-cache

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_ARRAY_JOB_ID="$SLURM_ARRAY_JOB_ID
echo "SLURM_ARRAY_TASK_ID="$SLURM_ARRAY_TASK_ID
echo "SLURM_JOB_NODELIST=" $SLURM_JOB_NODELIST

export CUDA_VISIBLE_DEVICES=${SLURM_ARRAY_TASK_ID}

all_gpus=1
processor='cpu'
dataset='cs'

cuda=${SLURM_ARRAY_TASK_ID}


singularity exec ~/data/containers/TensorFlow/tf-gpu.img ~/cpu/cs/FJ-1/File_Graph_Process_One_File_TF_IDF_Remove_Redundant.py ${SLURM_ARRAY_TASK_ID} $all_gpus $cuda $processor $dataset

#singularity exec ~/data/containers/TensorFlow/tf-gpu.img ~/File_Graph_Process_Internal_File_Sim2.py ${SLURM_ARRAY_TASK_ID} $all_gpus $cuda





