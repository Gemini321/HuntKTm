#!/bin/sh

export CUDA_VISIBLE_DEVICES='0,1,2,3'

# Quit cuda MPS if it's running
ps aux | grep nvidia-cuda-mps-control | grep -v grep > /dev/null
if [ $? -eq 0 ]; then
   echo quit | nvidia-cuda-mps-control
fi

# Enable persistent mode and MPS
nvidia-smi -pm 1
nvidia-cuda-mps-control -d

./run_experiments_ae.sh figure-3 # HuntKTm
# ./run_experiments_ae.sh figure-2 # HuntKT
# ./run_experiments_ae.sh figure-1 # HuntK

# cd ~/GPU-Sched/GPU-Sched/src/runtime/driver

# ./run_experiments_ae.sh figure-1 # CASE

# cp ~/GPU-Sched/GPU-Sched/src/runtime/driver/results/* ~/MultiGPU-Scheduler/driver/results

# cd ~/MultiGPU-Scheduler/driver
# python post_process_ae.py figure-1

# Disable persistent mode and MPS
echo quit | nvidia-cuda-mps-control
nvidia-smi -pm 0
