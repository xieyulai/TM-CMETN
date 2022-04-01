#!/bin/bash

PROCEDURE=train_prop

MODALITY=audio_video_text

AV_FUSION=add

AVT_FUSION=cat

EPOCH=20

BATCH=8

LR=1e-5

DATA_SELECT=2000

obj_coeff=1
noobj_coeff=1
reg_coeff=1
cen_coeff=0.01

#pretrained_cap_model_path=./checkpoint/train_cap/0325110315_audio_video_text_2000/best_cap_model.pt

DEVICE_IDS=0

python main_fcos.py --procedure $PROCEDURE  --modality $MODALITY  --epoch_num $EPOCH  --B $BATCH\
       --lr $LR --dout_p 0.1 --dataset_type $DATA_SELECT --dout_p_fcos 0.1\
       --device_ids $DEVICE_IDS --AV_fusion_mode $AV_FUSION --AVT_fusion_mode $AVT_FUSION\
       --obj_coeff $obj_coeff --noobj_coeff $noobj_coeff --reg_coeff $reg_coeff --cen_coeff $cen_coeff

