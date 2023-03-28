#!/bin/bash

. ./path.sh || exit 1;
export CUDA_VISIBLE_DEVICES=2


python main.py  --config /mnt/nvme-data1/waris/repo/vq-bnf-translator/conf/translator_vq128.yaml \
                --name=translator-vq128 \
                --ckpdir=/mnt/nvme-data1/waris/model_checkpoints/translator_vq \
                --seed=2 \