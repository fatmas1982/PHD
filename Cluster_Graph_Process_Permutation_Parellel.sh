#!/bin/bash --login
#BATCH --job-name=first-cuda-job
#SBATCH --partition=cpu
###SBATCH --gres=gpu:1
###SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=prmu_%A_.stdout
#SBATCH --out=prmu_%A.txt
#SBATCH --array=0-117
#SBATCH --error=prmu_%A.stderr
#SBATCH --export=ALL
#SBATCH --exclude=comp0[01-19]

#cat jobA.out >> jobB.out

module load CUDA/8.0.44

module load cuDNN/5.1-CUDA-8.0.44


module load Singularity/2.2.1

unset JAVA_HOME

export TEST_TMPDIR=~/data/bazel-cache


current_dir=$(pwd)
files=$(lfs find $current_dir/data/database/csv/semantics/sim)

echo 'files_size' ${#files[@]}

#for item in ${files[*]}
#do
    #printf "   %s\n" $item
     
#done



arr=(`echo $files | tr ' ' ' '`)
#echo 'arr' ${#arr[@]}
count=1
alltasks=`expr ${#arr[@]} - $count`
#echo 'All task' $alltasks


B=( "${arr[@]:1}" )
echo "${B[@]}" 



#for key in ${!B[*]}
#do 


export key=${SLURM_ARRAY_TASK_ID} #$key
#echo 'key' $key

export all_gpus=1
export processor='cpu'
export dataset='split'



step=4000
counter=`expr $key + $step`
 
echo 'Counter'$counter

export file=${B[$counter]}
  
export cuda=2 #$key
#printf "%4d: %s\n" $key ${B[$key]}
  #echo 'main_key '$key ' main_file ' ${arr[$key]}  >B_$key.out
#srun ~/Cluster_Graph_Process_Permutation.sh
singularity exec ~/data/containers/TensorFlow/tf-gpu.img ~/cpu/split/CJ-1/Cluster_Graph_Process_Permutation.py ${key} ${all_gpus} ${file} ${cuda} $processor $dataset

#done


#sacct -u helwan003u1 -j ${SLURM_JOB_ID} --format=User,JobID,Jobname,partition,state,elapsed,MaxVMSize,nodelist,reqmem,maxrss,averss,start,end



