#!/bin/bash

PROCEDURE=train_prop

MODALITY=audio_video_text

AV_FUSION=add

AVT_FUSION=cat

EPOCH=20

BATCH=1

LR=1e-5

DATA_SELECT=2000

#pretrained_cap_model_path=./checkpoint/train_cap/0815142747_audio_video_text/best_cap_model.pt

DEVICE_IDS=0

python main_fcos.py --procedure $PROCEDURE  --modality $MODALITY  --epoch_num $EPOCH  --B $BATCH\
       --lr $LR --dout_p 0.1 --dataset_type $DATA_SELECT --dout_p_fcos 0.1\
       --device_ids $DEVICE_IDS --AV_fusion_mode $AV_FUSION --AVT_fusion_mode $AVT_FUSION
