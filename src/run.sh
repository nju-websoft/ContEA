#!/usr/bin/env bash

datasets=('ZH-EN')
alphas=(0.1)  # better not change
betas=(0.1)   # better not change
g=0
Ms=(500)      # better not change
saved_path="../saved_model"
log_path="../logs"

for M in "${Ms[@]}"
do
for ds in "${datasets[@]}"
do
for alpha in "${alphas[@]}"
do
for beta in "${betas[@]}"
do

python main.py --batch 'base' \
              --gpu ${g} \
              --dataset ${ds} \
              --batch_size 1024 \
              --lr 0.0005 \
              --alpha 0.01 \
              --beta ${beta} \
              --save_path ${saved_path} \
              --log_path ${log_path} \
              --stop_step 3 \
              --M ${M}

baseModel=`ls ../saved_model/ | grep "${ds}_base" | tail -1`

python main.py --batch 'batch1' \
               --load_path ${baseModel} \
               --gpu ${g} \
               --dataset ${ds} \
               --batch_size 512 \
               --lr 0.01 \
               --save_path ${saved_path} \
               --log_path ${log_path} \
               --alpha ${alpha} \
               --beta ${beta} \
               --M ${M}

batch=(2 3 4 5)

for b in "${batch[@]}"
do
last=$((b-1))
lastModel=`ls ../saved_model/ | grep "${ds}_batch${last}" | tail -1`

python main.py \
       --batch "batch${b}" \
       --load_path ${lastModel} \
       --gpu ${g} \
       --save_path ${saved_path} \
       --log_path ${log_path} \
       --dataset ${ds} \
       --alpha ${alpha} \
       --beta ${beta} \
       --M ${M}

done
done
done
done
done