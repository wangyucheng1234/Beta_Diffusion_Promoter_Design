#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=ddsm_training       #Set the job name to "JobExample4"
#SBATCH --time=48:00:00              #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1  --cpus-per-task 28                  #Request 8 task
#SBATCH --mem=180G                  #Request 2560MB (2.5GB) per node
#SBATCH --output=beta_debug_mean.%j      #Send stdout/err to "Example4Out.[jobID]"
#SBATCH --gres=gpu:a100:2             #Request 1 "rtx" GPU per node
#SBATCH --partition=gpu              #Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=132747263243       #Set billing account to 123456
##SBATCH --mail-type=ALL              #Send email on all job events
##SBATCH --mail-user=wangyucheng@tamu.edu    #Send all emails to email_address 

#First Executable Line
cd $SCRATCH
# module load CUDA/9.2.88
module load Anaconda3
module load GCC/9.3.0
source activate ddsm
cd test
cd Beta_Diffusion_Promoter_Design
cd Beta_Diff_Promoter
cd promoter_design
python train_ori_beta_diff_promoter.py --sigmoid_power 0.1 --T 0 --eta 10000 --Scale 0.1 --Shift 0.8 --KLUB_Scale 0.9 --KLUB_Shift 0.09 --lossType KLUB --normalize_output