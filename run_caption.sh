#!/bin/bash

PROCEDURE=train_cap

MODALITY=audio_video_text

AV_FUSION=add

AVT_FUSION=cat

EPOCH=30

BATCH=2

LR=5e-5

one_by_one_starts_at=5

DATA_SELECT=2000

DEVICE_IDS=0

#pretrained_bmt_model_path=./checkpoint/best_cap_model.pt
#inherit_cap_model_path=./checkpoint/train_cap/best_cap_model.pt

python main_fcos.py --procedure $PROCEDURE  --modality $MODALITY  --epoch_num $EPOCH  --B $BATCH\
       --lr $LR --dataset_type $DATA_SELECT  --one_by_one_starts_at $one_by_one_starts_at\
       --device_ids $DEVICE_IDS --AV_fusion_mode $AV_FUSION --AVT_fusion_mode $AVT_FUSION
