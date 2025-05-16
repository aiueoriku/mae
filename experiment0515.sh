#!/bin/bash

set -e
# data_ids=(
#   10005 10075 10184 10387 10532 10737 11051
#   10017 10077 10240 10576 10760 11056
#   10028 10081 10289 10406 10590 10816 11057
#   10059 10093 10291 10444 10669 10852 11183
#   10061 10096 10345 10526 10671 10947
# )
data_ids=(
  10096 10532
)

for id in "${data_ids[@]}"; do
  python main_pretrain_cryoppp_particle_mask.py \
    --data_ids "$id" --batch_size 128 --lr 1e-5 --epochs 100 --mask_ratio 0.75 \

done

for id in "${data_ids[@]}"; do
  python main_pretrain_cryoppp_particle_mask.py \
    --data_ids "$id" --batch_size 128 --lr 1e-5 --epochs 100 --mask_ratio 0.5 \

done

for id in "${data_ids[@]}"; do
  python main_pretrain_cryoppp_particle_mask.py \
    --data_ids "$id" --batch_size 128 --lr 1e-5 --epochs 100 --mask_ratio 0.25 \

done

for id in "${data_ids[@]}"; do
  python main_pretrain_cryoppp_particle_mask.py \
    --data_ids "$id" --batch_size 128 --lr 1e-5 --epochs 100 --mask_ratio 0 \
    
done