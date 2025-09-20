#!/bin/bash

# Commands to run to perform the fine-tuning of HuBERT-ECG SMALL, BASE and LARGE models on the PTB-XL All dataset as done in the paper
# Standard train/val/test splits are used
# Altoigh previous commands actually reproduce the results we obtain on PTB-XL All, we consider this dataset only as an example
# More useful information available with:   python finetune.py --help

python finetune.py 1 ../processed_data/ptb_all_train.csv ../processed_data/ptb_all_val.csv 71 8 128 auroc \
    --load_path=hubert_ecg_base.pt \
    --training_steps=40000 --downsampling_factor=5 --label_start_index=4 --use_loss_weights \
    --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=500 \
    --finetuning_layerdrop=0.1 --random_crop --wandb_run_name=BASE_ptbxl_all


