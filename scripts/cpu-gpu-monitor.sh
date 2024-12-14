#!/bin/bash

while true
do
    top -n 1 -o +%CPU -b | head -n 17 | sed -n '8, 12{s/^ *//;s/ *$//;s/  */,/gp;};12q' >> cpu_utilization.dat
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv >> gpu_utilization.dat
    nvidia-smi --query-compute-apps=pid --format=csv,noheader >> gpu_by_process.dat
    sleep 1
done
