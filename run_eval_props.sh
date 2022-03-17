#!/bin/bash

PROCEDURE=evaluate

MODALITY=audio_video_text

pretrained_cap_path=./checkpoint/train_cap/0815142747_audio_video_text/best_cap_model.pt
prop_result_path=./log/train_prop/1204073452_audio_video_text/submissions/prop_results_val_1_e3_maxprop100.json

BATCH=64

DEVICE_IDS=0

python main_fcos.py --procedure $PROCEDURE  --modality $MODALITY\
            --pretrained_cap_model_path $pretrained_cap_path\
            --prop_pred_path $prop_result_path\
            --device_ids $DEVICE_IDS --B $BATCH

