#!/bin/bash

. ./path.sh || exit 1;
export CUDA_VISIBLE_DEVICES=3


python main.py  --config /mnt/nvme-data1/waris/repo/vq-bnf-translator/conf/translator_vq128_no_ctc.yaml \
                --name=translator-vq128-noctc \
                --ckpdir=/mnt/nvme-data1/waris/model_checkpoints/translator_vq \
                --seed=2 \