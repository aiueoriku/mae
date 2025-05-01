# !/bin/bash

### Line Masking

## input_file: "reconstruction.mrc", output_file: "grandmodel.mrc"
# default
python main_pretrain_cryoet.py \
  --lr 0.001 \
  --batch_size 32 \
  --mask_ratio 0.75 \
  --input_file "reconstruction.mrc" \
  --target_file "grandmodel.mrc" \
  --epochs 400

# mask_ratio 0.5
python main_pretrain_cryoet.py \
  --lr 0.001 \
  --batch_size 32 \
  --mask_ratio 0.5 \
  --input_file "reconstruction.mrc" \
  --target_file "grandmodel.mrc" \
  --epochs 400

# mask_ratio 0.25
python main_pretrain_cryoet.py \
  --lr 0.001 \
  --batch_size 32 \
  --mask_ratio 0.25 \
  --input_file "reconstruction.mrc" \
  --target_file "grandmodel.mrc" \
  --epochs 400

# mask_ratio 0.1
python main_pretrain_cryoet.py \
  --lr 0.001 \
  --batch_size 32 \
  --mask_ratio 0.1 \
  --input_file "reconstruction.mrc" \
  --target_file "grandmodel.mrc" \
  --epochs 400

## input_file: "grandmodel.mrc", output_file: "grandmodel.mrc"
# default
python main_pretrain_cryoet.py \
  --lr 0.001 \
  --batch_size 32 \
  --mask_ratio 0.75 \
  --input_file "grandmodel.mrc" \
  --target_file "grandmodel.mrc" \
  --epochs 400

# mask_ratio 0.5
python main_pretrain_cryoet.py \
  --lr 0.001 \
  --batch_size 32 \
  --mask_ratio 0.5 \
  --input_file "grandmodel.mrc" \
  --target_file "grandmodel.mrc" \
  --epochs 400

# mask_ratio 0.25
python main_pretrain_cryoet.py \
  --lr 0.001 \
  --batch_size 32 \
  --mask_ratio 0.25 \
  --input_file "grandmodel.mrc" \
  --target_file "grandmodel.mrc" \
  --epochs 400

# mask_ratio 0.1
python main_pretrain_cryoet.py \
  --lr 0.001 \
  --batch_size 32 \
  --mask_ratio 0.1 \
  --input_file "grandmodel.mrc" \
  --target_file "grandmodel.mrc" \
  --epochs 400

