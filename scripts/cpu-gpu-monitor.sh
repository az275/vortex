#!/bin/bash

mkdir data
cd data

while true
do
    { date +%s | tr -d '\n'; top -n 1 -o +%CPU -b | grep cascade_server | sed -e 's/\s\+/,/g'; } >> cpu_utilization.dat
    { date +%s | sed -z 's/\n/, /g'; nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv; } >> gpu_utilization.dat
    { date +%s | sed -z 's/\n/, /g'; nvidia-smi --query-compute-apps=process_name,pid,used_memory --format=csv,noheader; } >> gpu_by_process.dat
    sleep 3
done
