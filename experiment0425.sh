#!/bin/bash

### input_file: "reconstruction.mrc", output_file: "grandmodel.mrc"
# default
# CUDA_VISIBLE_DEVICES=0 \
#   python main_pretrain_cryoet.py \
#   --use_all_tomograms \
#   --lr 0.001 \
#   --batch_size 128 \
#   --mask_ratio 0.75 \
#   --input_file "reconstruction.mrc" \
#   --target_file "grandmodel.mrc" \
#   --epochs 100 &

# # mask_ratio 0.5
# CUDA_VISIBLE_DEVICES=1 \
#   python main_pretrain_cryoet.py \
#   --use_all_tomograms \
#   --lr 0.001 \
#   --batch_size 128 \
#   --mask_ratio 0.5 \
#   --input_file "reconstruction.mrc" \
#   --target_file "grandmodel.mrc" \
#   --epochs 100 &
# wait
# mask_ratio 0.25
# CUDA_VISIBLE_DEVICES=0 \
#   python main_pretrain_cryoet.py \
#   --use_all_tomograms \
#   --lr 0.001 \
#   --batch_size 128 \
#   --mask_ratio 0.25 \
#   --input_file "reconstruction.mrc" \
#   --target_file "grandmodel.mrc" \
#   --epochs 100 &

# # mask_ratio 0.1
# CUDA_VISIBLE_DEVICES=1 \
#   python main_pretrain_cryoet.py \
#   --use_all_tomograms \
#   --lr 0.001 \
#   --batch_size 128 \
#   --mask_ratio 0.1 \
#   --input_file "reconstruction.mrc" \
#   --target_file "grandmodel.mrc" \
#   --epochs 100 &

# wait

### input_file: "grandmodel.mrc", output_file: "grandmodel.mrc"
# default
# CUDA_VISIBLE_DEVICES=0 \
#   python main_pretrain_cryoet.py \
#   --use_all_tomograms \
#   --lr 0.001 \
#   --batch_size 128 \
#   --mask_ratio 0.75 \
#   --input_file "grandmodel.mrc" \
#   --target_file "grandmodel.mrc" \
#   --epochs 100 &

# # mask_ratio 0.5
# CUDA_VISIBLE_DEVICES=1 \
#   python main_pretrain_cryoet.py \
#   --use_all_tomograms \
#   --lr 0.001 \
#   --batch_size 128 \
#   --mask_ratio 0.5 \
#   --input_file "grandmodel.mrc" \
#   --target_file "grandmodel.mrc" \
#   --epochs 100 &
# wait
# mask_ratio 0.25
  python main_pretrain_cryoet.py \
  --use_all_tomograms \
  --lr 0.001 \
  --batch_size 128 \
  --mask_ratio 0.25 \
  --input_file "grandmodel.mrc" \
  --target_file "grandmodel.mrc" \
  --epochs 100 &

wait

# mask_ratio 0.1
  python main_pretrain_cryoet.py \
  --use_all_tomograms \
  --lr 0.001 \
  --batch_size 128 \
  --mask_ratio 0.1 \
  --input_file "grandmodel.mrc" \
  --target_file "grandmodel.mrc" \
  --epochs 100 &

