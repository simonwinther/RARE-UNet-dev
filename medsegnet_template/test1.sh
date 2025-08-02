#!/bin/bash

# Sweep 1: baseline (n-gate-blocks=8(3),16(4),32(5),64(6))
for i in {3..6}; do 
  tr wandb.log=true +architecture=msunet3d +dataset=Task04_Hippocampus \
    wandb.name=ms_instance_n-gate-blocks$((2**i)) \
    architecture.ablation.num_gate_blocks=$((2**i)) \
    gpu.devices=[2]
done

# Sweep 2: alpha=0.0
for i in {0..5}; do
  tr wandb.log=true +architecture=msunet3d +dataset=Task04_Hippocampus \
    wandb.name=ms_instance_a0_n-gate-blocks${i} \
    training.loss.params.alpha=0.0 \
    architecture.ablation.num_gate_blocks=${i} \
    gpu.devices=[2]
done

# Sweep 3: alpha=0.0 (n-gate-blocks=8(3),16(4),32(5),64(6), 128(7), 256(8), 512(9), 1024(10))
for i in {3..10}; do 
  tr wandb.log=true +architecture=msunet3d +dataset=Task04_Hippocampus \
    wandb.name=ms_instance_a0_n-gate-blocks$((2**i)) \
    training.loss.params.alpha=0.0 \
    architecture.ablation.num_gate_blocks=$((2**i)) \
    gpu.devices=[2]
done