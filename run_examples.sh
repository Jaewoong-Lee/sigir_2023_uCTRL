#!/bin/bash

python run_hyper.py --model=DirectAU --dataset=ml-1m --encoder=MF --params_file=search_hyper/directau.hyper --output_file=output_files/ml-1m/directau_mf --gpu_id=0 --eval=unbiased
python run_hyper.py --model=DirectAU --dataset=ml-1m --encoder=LightGCN --params_file=search_hyper/directau.hyper --output_file=output_files/ml-1m/directau_lightgcn --gpu_id=0 --eval=unbiased

python run_recbole.py --model=UCTRL --encoder=MF --dataset=ml-1m --weight_decay=0 --gamma=1 --gpu_id=0 --eval=unbiased &&
python run_recbole.py --model=UCTRL --encoder=LightGCN --dataset=ml-1m --weight_decay=0 --gamma=1 --n_layers=1 --gpu_id=0 --eval=unbiased &&
python run_recbole.py --model=UCTRL --encoder=MF --dataset=Gowalla2 --weight_decay=0 --gamma=1 --gpu_id=0 --eval=unbiased &&
python run_recbole.py --model=UCTRL --encoder=LightGCN --dataset=Gowalla2 --weight_decay=0 --gamma=1 --n_layers=1 --gpu_id=0 --eval=unbiased

python run_recbole.py --model=UCTRL --encoder=MF --dataset=Yelp2 --weight_decay=0 --gamma=1 --gpu_id=1 --eval=unbiased &&
python run_recbole.py --model=UCTRL --encoder=LightGCN --dataset=Yelp2 --weight_decay=0 --gamma=1 --n_layers=1 --gpu_id=1 --eval=unbiased

python run_recbole.py --model=UCTRL --encoder=MF --dataset=ml-1m --train_batch_size=512 --learning_rate=5e-3 --weight_decay=0 --gamma=1 --gpu_id=0 --eval=unbiased
python run_recbole.py --model=UCTRL --encoder=MF --dataset=Gowalla2  --train_batch_size=512 --learning_rate=5e-3 --weight_decay=0 --gamma=1 --gpu_id=0 --eval=unbiased
