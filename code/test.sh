#!/bin/bash

# Command to run to test HuBERT-ECG SMALL, BASE and LARGE models on the PTB-XL All test set
# More useful information with:   python test.py --help
# Finetuned models are available


python test.py ../processed_data/ptb_all_test.csv . 128 \
    models/checkpoints/supervised/hubert-5epo.pt \
    --downsampling_factor=5 \
    --label_start_index=4 \
    --tta \
    --save_id=ptb_all_base \
    --tta_aggregation=max
