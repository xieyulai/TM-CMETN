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


def train_cap_fcos(cfg):

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
        raise print('modality error!')

    # 损失函数定义
    criterion = LabelSmoothing(cfg.smoothing, train_dataset.pad_idx)

    # optimizer的选择
    optimizer = torch.optim.Adam(model.parameters(), cfg.lr)

    MILESTONES=cfg.milestones
    GAMMA=cfg.gamma
    print(type(cfg.lr), type(MILESTONES), type(GAMMA))

    # gamma = 0.2
    WARMUP_EPOCH=2
    warm_up_with_multistep_lr = lambda epoch: epoch / int(WARMUP_EPOCH) if epoch <= int(
        WARMUP_EPOCH) else GAMMA ** len([m for m in MILESTONES if m <= epoch])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_multistep_lr)

    if cfg.keep_train:
        path = cfg.inherit_cap_model_path
        cap_model_cpt = torch.load(path)
        # epoch_s = cap_model_cpt['epoch']
        weights = cap_model_cpt['model_state_dict']
        model_dict = model.state_dict()
        weights = {k.replace('module.', ''): v for k, v in weights.items() if k in model_dict}
        model.load_state_dict(weights)
        print('Keep train model load success!')

    # 训练模型的预处理
    print('\nModel Pre_processing!')
    model.to(torch.device(cfg.device))
    model = torch.nn.DataParallel(model, cfg.device_ids)

    param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total Number of Trainable Parameters: {param_num / 1000000} Mil.')

    if cfg.to_log:
        TBoard = tensorboard.SummaryWriter(log_dir=cfg.log_path)
        TBoard.add_scalar('debug/param_number', param_num, 0)
    else:
        TBoard = None

    best_epoch = 0
    best_metric = 0
    num_epoch_best_metric_unchanged = 0

    # 开始训练模型
    print('\nTrain Model!')
    for epoch in range(0, cfg.epoch_num):

        print(f'The best metrict was unchanged for {num_epoch_best_metric_unchanged} epochs.')
        print(f'Expected early stop @ {epoch + cfg.early_stop_after - num_epoch_best_metric_unchanged}')
        print(f'Started @ {cfg.curr_time}; Current timer: {timer(cfg.curr_time)}')

        # stop training if metric hasn't been changed for cfg.early_stop_after epochs
        if num_epoch_best_metric_unchanged == cfg.early_stop_after:
            break

        if cfg.modality == 'audio_video_text':
            multi_training_loop_fcos(cfg, model, train_loader, criterion, optimizer, epoch, TBoard)

        scheduler.step(epoch)

        # validation (1-by-1 word)
        if epoch >= cfg.one_by_one_starts_at:
            val_1_metrics = validation_1by1_loop(
                cfg, model, val_1_loader, greedy_decoder, epoch, TBoard,
            )
            val_2_metrics = validation_1by1_loop(
                cfg, model, val_2_loader, greedy_decoder, epoch, TBoard,
            )

            if cfg.to_log:
                metrics_avg = average_metrics_in_two_dicts(val_1_metrics, val_2_metrics)
                metrics_avg = metrics_avg['Average across tIoUs']

                TBoard.add_scalar('metrics/meteor', metrics_avg['METEOR'] * 100, epoch)
                TBoard.add_scalar('metrics/bleu4', metrics_avg['Bleu_4'] * 100, epoch)
                TBoard.add_scalar('metrics/bleu3', metrics_avg['Bleu_3'] * 100, epoch)
                # TBoard.add_scalar('metrics/bleu2', metrics_avg['Bleu_2'] * 100, epoch)
                # TBoard.add_scalar('metrics/bleu1', metrics_avg['Bleu_1'] * 100, epoch)
                # TBoard.add_scalar('metrics/rouge_l', metrics_avg['ROUGE_L'] * 100, epoch)
                # TBoard.add_scalar('metrics/cider', metrics_avg['CIDEr'] * 100, epoch)
                # TBoard.add_scalar('metrics/spice', metrics_avg['SPICE'] * 100, epoch)
                TBoard.add_scalar('metrics/precision', metrics_avg['Precision'] * 100, epoch)
                TBoard.add_scalar('metrics/recall', metrics_avg['Recall'] * 100, epoch)

                # saving the model if it is better than the best so far
                if best_metric < metrics_avg['METEOR']:
                    best_epoch = epoch
                    best_metric = metrics_avg['METEOR']

                    try:
                        save_model(cfg, epoch, model, optimizer, val_1_metrics, val_2_metrics, train_dataset.trg_voc_size)
                        # save_model(cfg, epoch, model, optimizer, val_1_metrics, train_dataset.trg_voc_size)
                    except Exception:
                        print('save model error!')
                    num_epoch_best_metric_unchanged = 0
                else:
                    num_epoch_best_metric_unchanged += 1

    print(f'{cfg.curr_time}')
    print(f'best_epoch:{best_epoch}---->best_metric: {best_metric}')
    if cfg.to_log:
        TBoard.close()
