#! /bin/bash
#BATCH --job-name=first-cuda-job
#SBATCH --partition=cpu
###SBATCH --gres=gpu:1
###SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1-3
###SBATCH --exclusive

#SBATCH --output=TFIDF_stdout_%A.stdout
#SBATCH --out=TFIDF_out_%A.txt

#SBATCH --error=TFIDF_stderr_%A.stderr
#SBATCH --export=ALL
#SBATCH --exclude=comp0[01-19]

module load CUDA/8.0.44
module load cuDNN/5.1-CUDA-8.0.44


module load Singularity/2.2.1

unset JAVA_HOME

export TEST_TMPDIR=~/data/bazel-cache

export CUDA_VISIBLE_DEVICES=${SLURM_ARRAY_TASK_ID}

all_gpus=1

processor='cpu'
dataset='cs'

cuda=${SLURM_ARRAY_TASK_ID}

singularity exec ~/data/containers/TensorFlow/tf-gpu.img ~/cpu/cs/CJ-2/Cluster_Graph_Process_TFIDF.py ${SLURM_ARRAY_TASK_ID} $all_gpus $cuda $processor $dataset



#sacct -u helwan003u1 -j ${SLURM_JOB_ID} --format=User,JobID,Jobname,partition,state,elapsed,MaxVMSize,nodelist,reqmem,maxrss,averss,start,end



