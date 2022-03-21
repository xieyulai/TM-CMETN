#!/bin/bash

PROCEDURE=evaluate

MODALITY=audio_video_text

pretrained_cap_path=./checkpoint/train_cap/0319091651_audio_video_text_2000/best_cap_model.pt
prop_result_path=./log/train_prop/0319091555_audio_video_text_2000/submissions/prop_results_val_1_e17_maxprop100.json

BATCH=64

DEVICE_IDS=0

python main_fcos.py --procedure $PROCEDURE  --modality $MODALITY\
            --pretrained_cap_model_path $pretrained_cap_path\
            --prop_pred_path $prop_result_path\
            --device_ids $DEVICE_IDS --B $BATCH

