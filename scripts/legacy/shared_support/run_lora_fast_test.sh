#!/bin/bash
# Fast speed test: no grad checkpointing, resolution 256, rank 8, 2 steps
python3 /Users/imds/Desktop/wbc_synthesis/scripts/legacy/shared_support/train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
  --instance_data_dir /Users/imds/Desktop/wbc_synthesis/data/processed/train_sharp/basophil \
  --output_dir /Users/imds/Desktop/wbc_synthesis/lora/weights/basophil_fast \
  --instance_prompt "microscopy image of a single basophil white blood cell, peripheral blood smear" \
  --resolution 256 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --learning_rate 5e-5 \
  --lr_scheduler cosine \
  --lr_warmup_steps 0 \
  --max_train_steps 2 \
  --rank 8 \
  --mixed_precision no \
  --seed 42 \
  --checkpointing_steps 9999 \
  --random_flip 2>&1
