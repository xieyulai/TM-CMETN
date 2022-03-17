import os

import numpy as np
import torch
import torch.nn.functional as F


def fill_missing_features_fcos(method, feature_size):
    if method == 'random':
        return torch.rand(1, feature_size)
    elif method == 'zero':
        return torch.zeros(1, feature_size).float()


# 根据proposal把一个完整视频的特征切割成为几部分
def crop_a_segment(feature, start, end, duration):
    S, D = feature.shape
    start_quantile = start / duration
    end_quantile = end / duration
    start_idx = int(S * start_quantile)
    end_idx = int(S * end_quantile)
    # handles the case when a segment is too small
    if start_idx == end_idx:
        # if the small segment occurs in the end of a video
        # [S:S] -> [S-1:S]
        if start_idx == S:
            start_idx -= 1
        # [S:S] -> [S:S+1]
        else:
            end_idx += 1
    feature = feature[start_idx:end_idx, :]

    if len(feature) == 0:
        return None
    else:
        return feature


def pad_segment(feature, max_feature_len, pad_idx):
    S, D = feature.shape
    assert S <= max_feature_len
    # pad
    l, r, t, b = 0, 0, 0, max_feature_len - S
    feature = F.pad(feature, [l, r, t, b], value=pad_idx)
    return feature


def load_features_from_npy_fcos(cfg, feature_names_list, video_id, start, end, duration,
                                pad_idx, phase, ids, get_full_feat=False):
    supported_feature_names = {'i3d_features', 'vggish_features', 'glove_features'}
    assert isinstance(feature_names_list, list)
    assert len(feature_names_list) > 0
    assert set(feature_names_list).issubset(supported_feature_names)

    stacks = {}
    if get_full_feat:
        stacks['orig_feat_length'] = {}

    # 根据proposal实现视频的每个proposal对应的特征片断堆栈
    if 'vggish_features' in feature_names_list:
        try:
            stack_vggish = np.load(os.path.join(cfg.audio_features_path, f'{video_id}.npy'))
            stack_vggish = torch.from_numpy(stack_vggish).float()

            if get_full_feat:
                stacks['orig_feat_length']['audio'] = stack_vggish.shape[0]
                if cfg.modality == 'audio_video_text':
                    stack_vggish = pad_segment(stack_vggish, cfg.pad_feats_up_to['audio'], pad_idx)   # 800
                else:
                    raise Exception
                    # stack_vggish = pad_segment(stack_vggish, cfg.pad_feats_up_to['audio'], pad_idx=0)
            else:
                stack_vggish = crop_a_segment(stack_vggish, start, end, duration)
        except FileNotFoundError:
            stack_vggish = None
        stacks['audio'] = stack_vggish
    # not elif
    if 'i3d_features' in feature_names_list:
        try:
            stack_rgb = np.load(os.path.join(cfg.video_features_path, f'{video_id}_rgb.npy'))
            stack_flow = np.load(os.path.join(cfg.video_features_path, f'{video_id}_flow.npy'))
            stack_rgb = torch.from_numpy(stack_rgb).float()
            stack_flow = torch.from_numpy(stack_flow).float()

            assert stack_rgb.shape == stack_flow.shape
            if get_full_feat:
                stacks['orig_feat_length']['rgb'] = stack_rgb.shape[0]
                stacks['orig_feat_length']['flow'] = stack_flow.shape[0]
                if cfg.modality == 'audio_video_text':
                    stack_rgb = pad_segment(stack_rgb, cfg.pad_feats_up_to['video'], pad_idx)      # 300
                    stack_flow = pad_segment(stack_rgb, cfg.pad_feats_up_to['video'], pad_idx=0)
                else:
                    raise Exception
                    # stack_rgb = pad_segment(stack_rgb, cfg.pad_feats_up_to['video'], pad_idx=0)
                    # stack_flow = pad_segment(stack_flow, cfg.pad_feats_up_to['video'], pad_idx=0)   # 300
            else:
                stack_rgb = crop_a_segment(stack_rgb, start, end, duration)
                stack_flow = crop_a_segment(stack_flow, start, end, duration)
        except FileNotFoundError:
            stack_rgb = None
            stack_flow = None
        stacks['rgb'] = stack_rgb
        stacks['flow'] = stack_flow
    if 'glove_features' in feature_names_list:
        try:
            stack_text_align = np.load(os.path.join(cfg.align_text_features_path, f'{video_id}.npy'))
            stack_text_align = torch.from_numpy(stack_text_align).float()
            stack_text = stack_text_align
            if get_full_feat:
                stacks['orig_feat_length']['text'] = stack_text_align.shape[0]
                if cfg.modality == 'audio_video_text':
                    stack_text = pad_segment(stack_text, cfg.pad_feats_up_to['text'], pad_idx)
                else:
                    raise Exception
            else:
                stack_text = crop_a_segment(stack_text, start, end, duration)
        except FileNotFoundError:
            if cfg.procedure == 'train_prop':
                stack_text = torch.zeros(2400, 300).float()
            else:
                stack_text = None

        stacks['text'] = stack_text
    if 'i3d_features' not in feature_names_list and 'vggish_features' not in feature_names_list and 'glove_features' not in feature_names_list:
        raise Exception(f'This methods is not implemented for {feature_names_list}')

    return stacks

