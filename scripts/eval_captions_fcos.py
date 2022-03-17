import pdb
import numpy as np
import torch
from torch.utils import tensorboard as tensorboard
from torch.utils.data import DataLoader

from datasets.captioning_dataset_fcos import ActivityNetCaptionsDatasetFCOS
from loss.label_smoothing import LabelSmoothing
from utilities.captioning_utils_fcos import average_metrics_in_two_dicts, timer
from epoch_loops.captioning_epoch_loops_fcos import (greedy_decoder, save_model,
                                                     multi_training_loop_fcos,
                                                     validation_1by1_loop,
                                                     )
from model.captioning_module_fcos import TriModalTransformer


def eval_caption(cfg):

    # doing our best to make it replicable
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(cfg.device_ids[0])

    # 数据集加载:带有batch的数据字典
    print('\nDataset Processing!')
    train_dataset = ActivityNetCaptionsDatasetFCOS(cfg, 'train', get_full_feat=False)
    val_1_dataset = ActivityNetCaptionsDatasetFCOS(cfg, 'val_1', get_full_feat=False)
    val_2_dataset = ActivityNetCaptionsDatasetFCOS(cfg, 'val_2', get_full_feat=False)

    print('\nDataset Loader!')
    train_loader = DataLoader(train_dataset, collate_fn=train_dataset.dont_collate)
    val_1_loader = DataLoader(val_1_dataset, collate_fn=val_1_dataset.dont_collate)
    val_2_loader = DataLoader(val_2_dataset, collate_fn=val_2_dataset.dont_collate)

    # 模型选择
    print('\nModel Select!')
    if cfg.modality == 'audio_video_text':
        model = TriModalTransformer(cfg, train_dataset)
    else:
        raise print('optim train modality!')

    cap_model_cpt = torch.load('./checkpoint/train_cap/0815142747_audio_video_text/best_cap_model.pt')
    weights = cap_model_cpt['model_state_dict']
    model_dict = model.state_dict()
    weights = {k.replace('module.', ''): v for k, v in weights.items() if k in model_dict}
    model.load_state_dict(weights)

    # 训练模型的预处理
    print('\nModel Pre_processing!')
    model.to(torch.device(cfg.device))
    model = torch.nn.DataParallel(model, cfg.device_ids)

    val_1_metrics = validation_1by1_loop(
        cfg, model, val_1_loader, greedy_decoder, 1000, None,
    )
    val_2_metrics = validation_1by1_loop(
        cfg, model, val_2_loader, greedy_decoder, 1000, None,
    )

    metrics_avg = average_metrics_in_two_dicts(val_1_metrics, val_2_metrics)
    metrics_avg = metrics_avg['Average across tIoUs']
    print(metrics_avg['METEOR'] * 100)
    print(metrics_avg['Bleu_4'] * 100)
    print(metrics_avg['Bleu_3'] * 100)
    print(metrics_avg['Precision'] * 100)
    print(metrics_avg['Recall'] * 100)
