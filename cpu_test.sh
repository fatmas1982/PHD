#! /bin/bash
#BATCH --job-name=first-cuda-job
###SBATCH --partition=gpu
###SBATCH --gres=gpu:1
#SBATCH --nodes=1
###SBATCH --ntasks=1
###SBATCH --cpus-per-task=1
#SBATCH --array=0-2
###SBATCH --exclusive

#SBATCH --output=one_File_Graph_Process_%A.stdout
#SBATCH --out=one_File_Graph_Process_%A.txt

#SBATCH --error=one_File_Graph_Process_%A.stderr
#SBATCH --export=ALL


#module load CUDA/8.0.44
#module load cuDNN/5.1-CUDA-8.0.44


module load Singularity/2.2.1

module load Python/3.5.2-intel-2016b

unset JAVA_HOME

export TEST_TMPDIR=~/data/bazel-cache

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_ARRAY_JOB_ID="$SLURM_ARRAY_JOB_ID
echo "SLURM_ARRAY_TASK_ID="$SLURM_ARRAY_TASK_ID
echo "SLURM_JOB_NODELIST=" $SLURM_JOB_NODELIST

export CUDA_VISIBLE_DEVICES=${SLURM_ARRAY_TASK_ID}

all_gpus=1

cuda=${SLURM_ARRAY_TASK_ID}


#singularity exec ~/data/containers/TensorFlow/tf-gpu.img ~/cpu_File_Graph_Process_One_File_TF_IDF_Remove_Redundant.py ${SLURM_ARRAY_TASK_ID} $all_gpus $cuda

singularity exec ~/data/containers/TensorFlow/tf-gpu.img ~/cpu_File_Graph_Process_One_File_TF_IDF_Remove_Redundant.py ${SLURM_ARRAY_TASK_ID} $all_gpus $cuda

#python3  ~/cpu_File_Graph_Process_One_File_TF_IDF_Remove_Redundant.py ${SLURM_ARRAY_TASK_ID} $all_gpus $cuda



