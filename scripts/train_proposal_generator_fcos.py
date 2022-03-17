import numpy as np
from torch.utils import tensorboard as tensorboard
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.captioning_dataset_fcos import ActivityNetCaptionsDatasetFCOS
from datasets.proposal_dataset_fcos import MultiProposalGenerationDatasetFCOS
from epoch_loops.proposal_epoch_loops_fcos import train_loop_fcos, validation_loop_fcos, train_av_loop_fcos, validation_av_loop_fcos
from model.proposal_generator_fcos import TrimodalProposalGeneratorFCOS
from utilities.captioning_utils_fcos import timer


def train_prop_fcos(cfg):
    # doing our best to make it replicable
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(cfg.device_ids[0])

    exp_name = cfg.curr_time[2:]

    print('\nDataset Processing!')
    train_dataset = ActivityNetCaptionsDatasetFCOS(cfg, 'train', get_full_feat=True)
    train_dataset = MultiProposalGenerationDatasetFCOS(cfg, 'train', train_dataset.pad_idx)
    valid_dataset = MultiProposalGenerationDatasetFCOS(cfg, 'val_1', train_dataset.pad_idx)

    print('\nDataset Loader!')
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True,
                              batch_size=cfg.train_batch_size,
                              collate_fn=train_dataset.collate4proposal_generation)

    valid_loader = DataLoader(valid_dataset, shuffle=False,
                              batch_size=cfg.inference_batch_size,
                              collate_fn=valid_dataset.collate4proposal_generation)

    # 模型的选择
    if cfg.modality == 'audio_video_text':
        model = TrimodalProposalGeneratorFCOS(cfg)
    else:
        raise print('modality error!')

    if cfg.keep_train:
        path = './best_model/train_prop/best_prop_model.pt'
        prop_model_cpt = torch.load(path)
        cfg = prop_model_cpt['config']
        best_metric = prop_model_cpt['best_metric']
        weights = prop_model_cpt['model_state_dict']
        weights = {k.replace('module.', ''): v for k, v in weights.items() if k in model.state_dict()}
        model.load_state_dict(weights)
        print('Keep train model load success!')

    model = model.to(cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), cfg.lr)

    MILESTONES=cfg.milestones
    GAMMA=cfg.gamma

    WARMUP_EPOCH=5
    warm_up_with_multistep_lr = lambda epoch: epoch / int(WARMUP_EPOCH*5) if epoch <= int(
        WARMUP_EPOCH) else GAMMA ** len([m for m in MILESTONES if m <= epoch])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_multistep_lr)

    # 训练前预处理操作
    param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total Number of Trainable Parameters: {param_num / 1000000} Mil.')

    if cfg.to_log:
        TBoard = tensorboard.SummaryWriter(log_dir=cfg.log_path)
        print(f'saving log @ {cfg.log_path}')
        TBoard.add_scalar('debug/param_number', param_num, 0)
    else:
        TBoard = None

    best_epoch = 0
    best_metric = 0
    num_epoch_best_metric_unchanged = 0

    # 开始训练
    print('\nTrain Model!')
    for epoch in range(0, cfg.epoch_num):
        print(f'The best metrict was unchanged for {num_epoch_best_metric_unchanged} epochs.')
        print(f'Expected early stop @ {epoch + cfg.early_stop_after - num_epoch_best_metric_unchanged}')
        print(f'Started @ {cfg.curr_time}; Current timer: {timer(cfg.curr_time)}')

        if num_epoch_best_metric_unchanged == cfg.early_stop_after:
            break

        if cfg.modality == 'audio_video_text':
            train_av_loop_fcos(cfg, model, optimizer, train_loader, epoch, TBoard)

            current_metric = validation_av_loop_fcos(
                cfg, model, optimizer, scheduler, valid_loader, epoch, best_metric, TBoard
            )
        else:
            raise print('modality error!')

        if current_metric > best_metric:
            best_epoch = epoch
            best_metric = current_metric
            num_epoch_best_metric_unchanged = 0
        else:
            num_epoch_best_metric_unchanged += 1

    print(f'best_epoch:{best_epoch}---->best_metric:{best_metric}')
    print(f'Experiment_validation: {exp_name}')
