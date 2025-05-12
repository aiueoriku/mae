#!/bin/bash
set -e

### Mask Ratio=0

## input_file: "reconstruction.mrc", output_file: "grandmodel.mrc"
python main_pretrain_cryoet.py \
  --lr 0.001 \
  --batch_size 32 \
  --mask_ratio 0 \
  --input_file "reconstruction.mrc" \
  --target_file "grandmodel.mrc" \
  --use_all_tomograms \
  --epochs 400

# lr 0.0001
python main_pretrain_cryoet.py \
  --lr 0.0001 \
  --batch_size 32 \
  --mask_ratio 0 \
  --input_file "reconstruction.mrc" \
  --target_file "grandmodel.mrc" \
  --use_all_tomograms \
  --epochs 400

## input_file: "grandmodel.mrc", output_file: "grandmodel.mrc"
# default
python main_pretrain_cryoet.py \
  --lr 0.001 \
  --batch_size 32 \
  --mask_ratio 0 \
  --input_file "grandmodel.mrc" \
  --target_file "grandmodel.mrc" \
  --use_all_tomograms \
  --epochs 400

# lr 0.0001
python main_pretrain_cryoet.py \
  --lr 0.0001 \
  --batch_size 32 \
  --mask_ratio 0 \
  --input_file "grandmodel.mrc" \
  --target_file "grandmodel.mrc" \
  --use_all_tomograms \
  --epochs 400

## input_file: "reconstruction.mrc", output_file: "reconstruction.mrc"
# default
python main_pretrain_cryoet.py \
  --lr 0.001 \
  --batch_size 32 \
  --mask_ratio 0 \
  --input_file "reconstruction.mrc" \
  --target_file "reconstruction.mrc" \
  --use_all_tomograms \
  --epochs 400

# lr 0.0001
python main_pretrain_cryoet.py \
  --lr 0.0001 \
  --batch_size 32 \
  --mask_ratio 0 \
  --input_file "reconstruction.mrc" \
  --target_file "reconstruction.mrc" \
  --use_all_tomograms \
  --epochs 400