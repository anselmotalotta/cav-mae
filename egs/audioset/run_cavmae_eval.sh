#!/bin/bash
##SBATCH -p sm
##SBATCH -x sls-sm-1,sls-2080-[1,3],sls-1080-[1],sls-sm-[5,6,7,12]
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=120000
#SBATCH --job-name="mae-as-ft"
#SBATCH --output=../log/%j_as_ft.txt

set -x
# comment this line if not running on sls cluster
. /data/sls/scratch/share-201907/slstoolchainrc
source /data/sls/scratch/yuangong/avbyol/venv-a5/bin/activate
export TORCH_HOME=../../pretrained_models

model=cav-mae-ft
ftmode=multimodal # or audioonly or videoonly

model_folder=/storage/models/cavmae
model_version=as-full-51.2 #as-20k-42.0
data_folder=/storage/data/cavmae

# you can replace with any checkpoint you want, but by default, we use cav-mae-scale++
cur_dir=$(pwd)
# wget -nc https://www.dropbox.com/s/l5t5geufdy3qvnv/audio_model.21.pth?dl=1 -O cav-mae-scale++.pth
pretrain_path=${model_folder}/${model_version}.pth


bal=bal
lr=1e-5
epoch=10
wa=True
wa_start=1
wa_end=10
dataset_mean=-5.081
dataset_std=4.4849
target_length=1024
noise=True
freqm=48
timem=192
mixup=0.5
batch_size=48
label_smooth=0.1

dataset=audioset

data_folder=/storage/data/cavmae
te_data=${data_folder}/audioset/audioset_eval_custom.json
label_csv=${data_folder}/audioset/class_labels_indices.csv

exp_dir=./exp/eval-${model}-version-${model_version}-${ftmode}-bs${batch_size}-r3
mkdir -p $exp_dir

CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run_cavmae_eval.py --model ${model} --dataset ${dataset} \
--data-val ${te_data}  --exp-dir $exp_dir \
--label-csv ${label_csv} --n_class 527 \
--n-epochs ${epoch} --batch-size $batch_size --save_model True \
--label_smooth ${label_smooth} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --noise ${noise} \
--metrics mAP --warmup True \
--wa ${wa} --wa_start ${wa_start} --wa_end ${wa_end} \
--pretrain_path ${pretrain_path} --ftmode ${ftmode} \
--num-workers 32