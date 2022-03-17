import os
import json
from time import time
import torch

from utilities.captioning_utils_fcos import HiddenPrints
from epoch_loops.captioning_epoch_loops_fcos import calculate_metrics
# from sklearn.cluster import KMeans


def tiou_vectorized(segments1, segments2, without_center_coords=False, center_length=True):

    def center_length_2_start_end(segments):
        '''there is get_corner_coords(predictions) and has a bit diffrenrent logic. both are kept'''
        # print('segments[:, 0]:\n', segments[:, 0])
        # print('segments[:, 1]:\n', segments[:, 1])
        start = segments[:, 0] - segments[:, 1] / 2
        end = segments[:, 0] + segments[:, 1] / 2
        return start, end

    # add 'fake' center coordinates. You can use any value, we use zeros
    if without_center_coords:
        segments1 = torch.cat([torch.zeros_like(segments1), segments1], dim=1) # 在第1维度与全0矩阵拼接起来
        segments2 = torch.cat([torch.zeros_like(segments2), segments2], dim=1)

    M, D = segments1.shape   # (48,2)
    N, D = segments2.shape   # (50,2)

    # TODO: replace with get_corner_coords from localization_utils
    if center_length:
        start1, end1 = center_length_2_start_end(segments1)
        start2, end2 = center_length_2_start_end(segments2)
    else:
        start1, end1 = segments1[:, 0], segments1[:, 1]
        start2, end2 = segments2[:, 0], segments2[:, 1]

    # broadcasting
    start1 = start1.view(M, 1)
    end1 = end1.view(M, 1)
    start2 = start2.view(1, N)
    end2 = end2.view(1, N)

    # calculate segments for intersection
    intersection_start = torch.max(start1, start2)    # (M,N) (48,50)
    intersection_end = torch.min(end1, end2)          # (M,N) (48,50)

    # we make sure that the area is 0 if size of a side is negative
    # which means that intersection_start > intersection_end which is not feasible
    # Note: adding one because the coordinates starts at 0 and let's
    intersection = torch.clamp(intersection_end - intersection_start, min=0.0)

    # finally we calculate union for each pair of segments
    union1 = (end1 - start1)
    union2 = (end2 - start2)
    union = union1 + union2 - intersection    # (M,1)+(1,N)-(M,N)
    # print('union":\n', union)
    union = torch.min(torch.max(end1, end2) - torch.min(start1, start2), union)
    # print('union1":\n', union)

    tious = intersection / (union + 1e-8)    # tious的计算公式
    # print('tious:\n', tious)
    return tious


def tiou_vectorized_fcos(segments1, segments2, without_center_coords=False):

    # add 'fake' center coordinates. You can use any value, we use zeros
    if without_center_coords:
        segments1 = torch.cat([torch.zeros_like(segments1), segments1], dim=1) # 在第1维度与全0矩阵拼接起来
        segments2 = torch.cat([torch.zeros_like(segments2), segments2], dim=1)

    M, D = segments1.shape   # (1,5)
    N, D = segments2.shape   # (num_points-1,5)

    # TODO: replace with get_corner_coords from localization_utils
    start1, end1 = segments1[:, 0], segments1[:, 1]
    start2, end2 = segments2[:, 0], segments2[:, 1]

    # broadcasting
    start1 = start1.view(M, 1)
    end1 = end1.view(M, 1)
    start2 = start2.view(1, N)
    end2 = end2.view(1, N)

    # calculate segments for intersection
    intersection_start = torch.max(start1, start2)    # (M,N) (1,num_points-1)
    intersection_end = torch.min(end1, end2)          # (M,N) (1,num_points-1) (48,50)

    # we make sure that the area is 0 if size of a side is negative
    # which means that intersection_start > intersection_end which is not feasible
    # Note: adding one because the coordinates starts at 0 and let's
    intersection = torch.clamp(intersection_end - intersection_start, min=0.0)

    # finally we calculate union for each pair of segments
    union1 = (end1 - start1)
    union2 = (end2 - start2)
    union = union1 + union2 - intersection    # (M,1)+(1,N)-(M,N)
    # print('union":\n', union)
    union = torch.min(torch.max(end1, end2) - torch.min(start1, start2), union)
    # print('union1":\n', union)

    tious = intersection / (union + 1e-8)    # tious的计算公式
    # print('tious:\n', tious)
    return tious


def calculate_f1(recall, precision):
    f1 = 2*recall*precision / (recall + precision + 1e-16)
    return f1


def filter_meta_for_video_id(meta, video_id, column_name='video_id'):
    return meta[meta[column_name] == video_id]    # 把meta中video_id与所给的video_idx相同的数据输出来


def get_corner_coords(predictions):
    '''predictions (B, S*A, num_feats)'''
    starts = predictions[:, :, 3] - predictions[:, :, 0]
    ends = predictions[:, :, 3] + predictions[:, :, 1]
    predictions[:, :, 0] = starts
    predictions[:, :, 1] = ends
    return predictions


def add_dict_to_another_dict(one_dict, another_dict):
    another_dict = {k: another_dict.get(k, 0) + v for k, v in one_dict.items()}
    return another_dict


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def select_topk_predictions(model_output, k):
    '''model_output (B, points_num, num_feats)'''
    B, P, num_feats = model_output.shape
    # sort model_output on confidence score (2nd col) within each batch
    # (B, S) <-
    output = torch.sqrt(model_output[:, :, 4] * model_output[:, :, 2])
    indices = output.argsort(descending=True)
    # (B, S, 1) <- .view()
    # (B, S, num_feats) <- .repe
    # at()
    indices = indices.view(B, P, 1).repeat(1, 1, num_feats)
    model_output = model_output.gather(1, indices)
    # select top k
    # (B, k, num_feats) <-
    model_output = model_output[:, :k, :]
    return model_output


def trim_proposals(model_output, duration_in_secs):
    '''Changes in-place model_output (B, points_num, num_feats), starts & ends are in seconds'''
    # for broadcasting it for batches
    duration_in_secs = torch.tensor(duration_in_secs, device=model_output.device).view(-1, 1)
    min_start = torch.tensor([0.0], device=model_output.device)
    # clip start for negative values and if start is longer than the duration
    model_output[:, :, 0] = model_output[:, :, 0].max(min_start).min(duration_in_secs)
    # clip end
    model_output[:, :, 1] = model_output[:, :, 1].min(duration_in_secs)
    return model_output


def remove_very_short_segments(model_output, shortest_segment_prior):
    model_output = model_output
    # (1, A*S) <-
    lengths = model_output[:, :, 1] - model_output[:, :, 0]
    # (A*S) <-
    lengths.squeeze_()
    # (A*S)
    model_output = model_output[:, lengths > shortest_segment_prior, :]

    return model_output


def non_max_suppresion(video_preds, tIoU_threshold):
    '''video_preds (AS, num_features)'''
    # model_output should be sorted according to conf_score, otherwise sort it here
    model_output_after_nms = []
    while len(video_preds) > 0:
        # (1, num_feats) <- (one_vid_pred[0, :].unsqueeze(0))
        model_output_after_nms.append(video_preds[0, :].unsqueeze(0))
        if len(video_preds) == 1:
            break
        # (1, *) <- (1, num_feats) x (*, num_feats)
        tious = tiou_vectorized_fcos(video_preds[0, :].unsqueeze(0), video_preds[1:, :])
        # (*) <- (1, *)
        tious = tious.reshape(-1)
        # (*', num_feats)
        video_preds = video_preds[1:, :][tious < tIoU_threshold]
    # (new_N, D) <- a list of (1, num_feats)
    model_output = torch.cat(model_output_after_nms)
    return model_output


def postprocess_preds(model_output, cfg, batch):
    '''
        model_output (B, points_num, num_features) with l & r & c & p in second
        1. Takes top-[max_prop_per_vid] predictions
        3. Converts l & r into start & end
        4. Trims the segments according to sanity and original duration
    '''
    # select top-[max_prop_per_vid] predictions
    # (B, points_num, num_feats) <- (B, points_num, num_feats)
    model_output = select_topk_predictions(model_output, k=cfg.max_prop_per_vid)
    # (B, points_num, num_feats) <- (B, points_num, num_feats)
    model_output = get_corner_coords(model_output)
    # clip start & end to duration
    # (B, points_num, num_feats) <- (B, points_num, num_feats)
    model_output = trim_proposals(model_output, batch['duration_in_secs'])
    # (B, points_num, num_feats) <-
    return model_output


class AnetPredictions(object):

    def __init__(self, cfg, phase, epoch):
        self.predictions = {
            'version': 'VERSION 1.0',
            'external_data': {
                'used': True,
                'details': ''
            },
            'results': {}
        }
        self.phase = phase
        self.epoch = epoch
        self.cfg = cfg
        self.segments_used = 0        # 经进一步处理以后，剩余片段的总数
        self.segments_total = 0       # 选取top100所得到的片段总数
        self.num_vid_w_no_props = 0

        # if cfg.modality == 'audio_video_text':
        # if phase == 'val_1':
        #     self.reference_paths = [cfg.val_reference_paths[0]]
        # elif phase == 'val_2':
        #     self.reference_paths = [cfg.val_reference_paths[1]]
        # else:
        #     if phase == 'val_1':
        #         self.reference_paths = [cfg.reference_paths[0]]
        #         # tIoUs = [0.5]  # no need to wait: they all the same as they are predicted for gt segments
        #     elif phase == 'val_2':
        #         self.reference_paths = [cfg.reference_paths[1]]

    def add_new_predictions(self, model_output, batch):
        '''
        model_output (B,　points_num, num_features)
        updates anet_prediction dict with the predictions from model_output
        '''
        model_output = postprocess_preds(model_output, self.cfg, batch)

        B, P1, D = model_output.shape
        num_of_props_written = 0

        shortest_segment_prior = 0.2  # (sec)
        # 对每一个视频的proposal的结果进行处理
        for b, video_preds in enumerate(model_output):
            vid_id = batch['video_ids'][b]
            vid_id_preds = []

            if self.cfg.nms_tiou_thresh is not None:
                # (nms_N, num_features)<- (points_num, num_features)
                video_preds = non_max_suppresion(video_preds, self.cfg.nms_tiou_thresh)

            for pred_start, pred_end, _, _, pred_conf in video_preds.tolist():
                segment = {}
                start, end = round(pred_start, 5), round(pred_end, 5)
                if end - start > shortest_segment_prior:
                    segment['sentence'] = ''
                    segment['proposal_score'] = round(pred_conf, 5),
                    segment['timestamp'] = [start, end]
                    vid_id_preds.append(segment)
                    num_of_props_written += 1
            # sometimes all segmets are removed as they are too short. Hence, the preds are saved
            # only if  at least one segment was added to predictions
            if len(vid_id_preds) > 0:
                self.predictions['results'][vid_id] = vid_id_preds
            else:
                # print(f'{vid_id} has empty proposal list')
                self.num_vid_w_no_props += 1

        self.segments_total += B * P1
        self.segments_used += num_of_props_written

        # 针对此batch来说，每个视频提取proposal的平均数量
        num_of_props_written_per_video = num_of_props_written / B
        return num_of_props_written_per_video

    def write_anet_predictions_to_json(self):
        # save only val_1 because the props are the same. 1 not the 2 one because 1st has +30 vids
        if self.phase == 'val_1':
            submission_folder = os.path.join(self.cfg.log_path, 'submissions')
            filename = f'prop_results_{self.phase}_e{self.epoch}_maxprop{self.cfg.max_prop_per_vid}.json'
            self.submission_path = os.path.join(submission_folder, filename)
            # 以递归的方式创建一个目录
            os.makedirs(submission_folder, exist_ok=True)
            # if the same file name already exists, append random num to the path
            if os.path.exists(self.submission_path):
                self.submission_path = self.submission_path.replace('.json', f'_{time()}.json')
            with open(self.submission_path, 'w') as outf:
                json.dump(self.predictions, outf)
        elif self.phase == 'train':
            submission_folder = os.path.join(self.cfg.log_path, 'submissions')
            filename = f'prop_results_{self.phase}_e{self.epoch}_maxprop{self.cfg.max_prop_per_vid}.json'
            self.submission_path = os.path.join(submission_folder, filename)
            # 以递归的方式创建一个目录
            os.makedirs(submission_folder, exist_ok=True)
            # if the same file name already exists, append random num to the path
            if os.path.exists(self.submission_path):
                self.submission_path = self.submission_path.replace('.json', f'_{time()}.json')
            with open(self.submission_path, 'w') as outf:
                json.dump(self.predictions, outf)
        else:
            raise NotImplementedError

    def evaluate_predictions(self):
        # 对于全部的视频来说，每个视频提取的proposal的平均数量
        print(f'{self.cfg.max_prop_per_vid*self.segments_used/self.segments_total:.2f} props/vid')
        # during first epochs we have empty preds because we apply postprocess_preds() on preds
        if self.num_vid_w_no_props > 0:
            print(f'Number of videos with no proposals: {self.num_vid_w_no_props}')
        # blocks the printing
        with HiddenPrints():
            # metrics = calculate_metrics(
            #     self.cfg.reference_paths, self.submission_path, self.cfg.tIoUs,
            #     self.cfg.max_prop_per_vid, verbose=True, only_proposals=True
            # )
            metrics = calculate_metrics(
                # self.reference_paths,
                self.cfg.val_reference_paths,
                self.submission_path, self.cfg.tIoUs,
                self.cfg.max_prop_per_vid, verbose=True, only_proposals=True
            )
        return metrics
