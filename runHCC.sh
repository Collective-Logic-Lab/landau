#!/bin/bash

## runHCC.sh
##
## Bryan Daniels
## 2023/3/31 branched from runFittingAnalysis.sh
## 2021/9/27 branched from runLandauTestSimulations.sh
## 2021/8/4 branched from run_control_kernels.sh
## 2.19.2019 branched from RunScan_BCD.sh
## 10.10.2018 branched from runWormFitting.sh
## 7.18.2018 branched from runWormFitting.sub
## 9.10.2015
## 3.11.2016
## 3.22.2016
## 4.4.2016
## 

#SBATCH -p fn1 # fn1 = high-memory partition
#SBATCH -n 4 #40 #20       # number of cores
#SBATCH -t 1-00:00 # wall time (D-HH:MM)
#SBATCH -o slurm.landau_HCC.%A_%a.out
#SBATCH -e slurm.landau_HCC.%A_%a.err
#SBATCH -q normal

#module load anaconda/2.1.0
#module load gcc
#module load openmpi

#module load openmpi/1.5.5/gcc.4.7.2
#module load anaconda/2.1.0

# 7.18.2018
#module load openmpi/3.0.0-gcc-7.3.0
# try older version of openmpi
#module load openmpi/1.10.2-gnu-4.9.3i
#module load openmpi/3.0.0-gcc-4.9.4
#module load anaconda2/4.4.0
#try older version of anaconda
#module load anaconda2/4.2.0

# 7.20.2018 try again with my own installed
# version of anaconda2
#module load openmpi/3.0.0-gcc-7.3.0

# 7.20.2018 try again again with my own
# installed version of anaconda 2
# (don't forget to completely reinstall
#  pypar)
#module load openmpi/2.1.1-intel-2017x

module load anaconda/py3

module load mathematica/13.1

#export PATH="/home/bdaniel6/anaconda2/bin:$PATH"
#export PYTHONPATH=$PYTHONPATH:~/lib/python2.7/site-packages/
#export MPLBACKEND=Agg

cd ~/landau/

# 2020.8.18 trying to use the python3 environment in conda 
# (seems harder than it should be)

## >>> conda initialize >>>
## !! Contents within this block are managed by 'conda init' !!
#__conda_setup="$('/home/bdaniel6/anaconda2/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
#if [ $? -eq 0 ]; then
#    eval "$__conda_setup"
#else
#    if [ -f "/home/bdaniel6/anaconda2/etc/profile.d/conda.sh" ]; then
#        . "/home/bdaniel6/anaconda2/etc/profile.d/conda.sh"
#    else
#        export PATH="/home/bdaniel6/anaconda2/bin:$PATH"
#    fi
#fi
#unset __conda_setup
## <<< conda initialize <<<
#
#conda activate python3

python ~/landau/hepatocellular-carcinoma-data.py


