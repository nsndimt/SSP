Code For Less is more: A Prototypical framework for Efficient Few-Shot Named Entity Recognition


## Enviroment
- pytorch-lightning                  1.6.0
- torch                              1.11.0
- transformers                       4.6.1

## Dataset
### FewNERD
```
git clone https://github.com/thunlp/Few-NERD.git
cd Few-NERD
bash data/download.sh supervised
bash data/download.sh episode-data
unzip -d data/ data/episode-data.zip
```
### SNIPS
```
git clone https://github.com/AtmaHou/FewShotTagging.git
cd FewShotTagging
wget https://atmahou.github.io/attachments/ACL2020data.zip
unzip ACL2020data.zip
```
## Run

### FewNERD
create `seed_fewnerd.sh`
```
#!/bin/bash
mode=$1
model=$2
seed=$RANDOM
mode=$2
seed=$RANDOM
if [ $model = "proto" ]
then
python reproduce.py --wandb_proj FewNERD_seed  --model proto --mode $mode --N 5  --K 1 --batch_size 16 --seed $seed
python reproduce.py --wandb_proj FewNERD_seed  --model proto --mode $mode --N 5  --K 5 --batch_size 4  --seed $seed
python reproduce.py --wandb_proj FewNERD_seed  --model proto --mode $mode --N 10 --K 1 --batch_size 8  --seed $seed
python reproduce.py --wandb_proj FewNERD_seed  --model proto --mode $mode --N 10 --K 5 --batch_size 2  --seed $seed
fi
if [ $model = "nnshot" ]
then
python reproduce.py --wandb_proj FewNERD_seed  --model nnshot --mode $mode --N 5  --K 1 --batch_size 16 --seed $seed --dropout 0.1
python reproduce.py --wandb_proj FewNERD_seed  --model nnshot --mode $mode --N 5  --K 5 --batch_size 4  --seed $seed --dropout 0.1
python reproduce.py --wandb_proj FewNERD_seed  --model nnshot --mode $mode --N 10 --K 1 --batch_size 8  --seed $seed --dropout 0.1
python reproduce.py --wandb_proj FewNERD_seed  --model nnshot --mode $mode --N 10 --K 5 --batch_size 2  --seed $seed --dropout 0.1
fi
if [ $model = "structshot" ]
then
python reproduce.py --wandb_proj FewNERD_seed  --model structshot --mode $mode --N 5  --K 1 --batch_size 16 --seed $seed --dropout 0.1 --tau 0.32
python reproduce.py --wandb_proj FewNERD_seed  --model structshot --mode $mode --N 5  --K 5 --batch_size 4  --seed $seed --dropout 0.1 --tau 0.434
python reproduce.py --wandb_proj FewNERD_seed  --model structshot --mode $mode --N 10 --K 1 --batch_size 8  --seed $seed --dropout 0.1 --tau 0.32
python reproduce.py --wandb_proj FewNERD_seed  --model structshot --mode $mode --N 10 --K 5 --batch_size 2  --seed $seed --dropout 0.1 --tau 0.434
fi
if [ $model = "span" ]
then
checkpoint_path="seed_checkpoint/span_"$mode"_seed_"$seed
while [ -d "$checkpoint_path" ]
do
checkpoint_path="seed_checkpoint/span_"$mode"_seed_"$seed
done
python pretrain.py --wandb_proj FewNERD_seed --model span --mode $mode --train_step 6000 --val_interval 500 --backbone_name bert-base-uncased --output_dir_overwrite --output_dir $checkpoint_path --seed $seed
python train.py --wandb_proj FewNERD_seed --model span --mode $mode --N 5  --K 5 --batch_size 4  --train_step 6000 --val_interval 500 --lr 5e-4 --wd 0.01 --dropout 0.1 --pretrained_encoder $checkpoint_path/encoder.state --backbone_name bert-base-uncased --use_atten --seed $seed
python train.py --wandb_proj FewNERD_seed --model span --mode $mode --N 5  --K 1 --batch_size 16 --train_step 6000 --val_interval 500 --lr 5e-4 --wd 0.01 --dropout 0.1 --pretrained_encoder $checkpoint_path/encoder.state --backbone_name bert-base-uncased --use_atten --seed $seed
python train.py --wandb_proj FewNERD_seed --model span --mode $mode --N 10 --K 5 --batch_size 2  --train_step 6000 --val_interval 500 --lr 5e-4 --wd 0.01 --dropout 0.1 --pretrained_encoder $checkpoint_path/encoder.state --backbone_name bert-base-uncased --use_atten --seed $seed
python train.py --wandb_proj FewNERD_seed --model span --mode $mode --N 10 --K 1 --batch_size 8  --train_step 6000 --val_interval 500 --lr 5e-4 --wd 0.01 --dropout 0.1 --pretrained_encoder $checkpoint_path/encoder.state --backbone_name bert-base-uncased --use_atten --seed $seed
fi
```
run `bash seed_fewnerd.sh inter span` to run four expriments on Few-NERD inter split using our model
### SNIPS
create `seed_snips.sh`
```
#!/bin/bash
model=$1
seed=$RANDOM
for (( i=1; i<8; i++ ))
do
  if [ $model = "structshot" ];
  then
  python snips_train.py --wandb_proj SNIPS_seed  --model $model --tau 0.32 --cv_id $i  --K 1 --train_epoch 20 --bert_lr 5e-5 --seed $seed
  python snips_train.py --wandb_proj SNIPS_seed  --model $model --tau 0.434 --cv_id $i  --K 5 --train_epoch 20 --bert_lr 5e-5 --seed $seed
  elif [ $model = "span" ];
  then
  python snips_train.py --wandb_proj SNIPS_seed  --model span --cv_id $i  --K 1 --train_epoch 40 --bert_lr 5e-5 --seed $seed --dropout 0.1 --wd 0.01 --use_atten --neg_rate 1.4
  python snips_train.py --wandb_proj SNIPS_seed  --model span --cv_id $i  --K 5 --train_epoch 40 --bert_lr 5e-5 --seed $seed --dropout 0.1 --wd 0.01 --use_atten --neg_rate 1.4
  else
  python snips_train.py --wandb_proj SNIPS_seed  --model $model --cv_id $i  --K 1 --train_epoch 20 --bert_lr 5e-5 --seed $seed
  python snips_train.py --wandb_proj SNIPS_seed  --model $model --cv_id $i  --K 5 --train_epoch 20 --bert_lr 5e-5 --seed $seed
  echo "not structshot"
  fi
done
```
run `bash seed_snips.sh span` to run all expriments on the 7 domain of SNIPS with both 1-shot and 5-shot using our model


