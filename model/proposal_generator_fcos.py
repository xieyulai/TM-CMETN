import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import json

from model.encoders_fcos import TriModalEncoder, BiModalEncoderOne
from model.blocks import FeatureEmbedder, Identity, PositionalEncoder


class BackBone(nn.Module):

    def __init__(self, C2_inplanes, C3_inplanes, C4_inplanes, C5_inplanes):
        super(BackBone, self).__init__()
        # 2048-->1024
        self.C3_1 = nn.Sequential(
            Transpose(),
            nn.LayerNorm(C2_inplanes),
            Transpose(),
            nn.Conv1d(C2_inplanes,
                      C3_inplanes,
                      kernel_size=2,
                      stride=2,
                      padding=0),
        )
        # 1024-->1024
        self.C4_1 = nn.Sequential(
            Transpose(),
            nn.LayerNorm(C3_inplanes),
            Transpose(),
            nn.Conv1d(C3_inplanes,
                      C4_inplanes,
                      kernel_size=2,
                      stride=2,
                      padding=0),
        )
        # 1024-->1024
        self.C5_1 = nn.Sequential(
            Transpose(),
            nn.LayerNorm(C4_inplanes),
            Transpose(),
            nn.Conv1d(C4_inplanes,
                      C5_inplanes,
                      kernel_size=2,
                      stride=2,
                      padding=0),
        )

    def forward(self, inputs):
        C3 = self.C3_1(inputs)
        C4 = self.C4_1(C3)
        C5 = self.C5_1(C4)

        return [C3, C4, C5]


class FPN(nn.Module):
    # 改进的特征金字塔网络，共８层
    def __init__(self, C3_inplanes, C4_inplanes, C5_inplanes, planes, use_p5=False):
        super(FPN, self).__init__()
        self.use_p5 = use_p5
        self.F3 = nn.Conv1d(C3_inplanes,
                            planes,
                            kernel_size=1,
                            stride=1,
                            padding=0)
        self.P3 = nn.Conv1d(planes,
                            planes,
                            kernel_size=3,
                            stride=1,
                            padding=1)
        self.F4 = nn.Conv1d(C4_inplanes,
                            planes,
                            kernel_size=1,
                            stride=1,
                            padding=0)
        self.P4 = nn.Conv1d(planes,
                            planes,
                            kernel_size=3,
                            stride=1,
                            padding=1)
        self.F5 = nn.Conv1d(C5_inplanes,
                            planes,
                            kernel_size=1,
                            stride=1,
                            padding=0)
        self.P5 = nn.Conv1d(planes,
                            planes,
                            kernel_size=3,
                            stride=1,
                            padding=1)
        self.P6 = nn.Conv1d(planes,
                            planes,
                            kernel_size=3,
                            stride=2,
                            padding=1)

        self.P7 = nn.Conv1d(planes, planes, kernel_size=3, stride=2, padding=1)
        self.P8 = nn.Conv1d(planes, planes, kernel_size=3, stride=2, padding=1)
        self.P9 = nn.Conv1d(planes, planes, kernel_size=3, stride=2, padding=1)
        self.P10 = nn.Conv1d(planes, planes, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        [C3, C4, C5] = inputs

        P5 = self.F5(C5)
        P4 = self.F4(C4)
        P4 = F.interpolate(P5, size=(P4.shape[2]),
                           mode='nearest') + P4
        P3 = self.F3(C3)
        P3 = F.interpolate(P4, size=(P3.shape[2]),
                           mode='nearest') + P3

        P5 = self.P5(P5)
        P4 = self.P4(P4)
        P3 = self.P3(P3)
        del C3, C4, C5

        P6 = self.P6(P5)
        P7 = self.P7(P6)
        P8 = self.P8(P7)
        P9 = self.P9(P8)
        P10 = self.P10(P9)

        return [P3, P4, P5, P6, P7, P8, P9, P10]


class Transpose(nn.Module):
    """
        LayerNorm expects (B, S, D) but receives (B, D, S)
        Conv1d expects (B, D, S) but receives (B, S, D)
    """

    def __init__(self):
        super(Transpose, self).__init__()

    def forward(self, x):
        return x.permute(0, 2, 1)


class FCOSRegCenterClsHead(nn.Module):
    # 参照FCOS论文中后续的改进方案，将centerness heads与回归heads共用。
    def __init__(self, layer_norm, inplanes, dout_p, num_layers=4, prior=0.01):
        super(FCOSRegCenterClsHead, self).__init__()
        cls_branch_layers = []
        reg_branch_layers = []

        # 将reg与center放在一个分支，cls放在另一个分支
        for _ in range(num_layers):

            if layer_norm:
                cls_branch_layers.append(Transpose())
                cls_branch_layers.append(nn.LayerNorm(inplanes))
                cls_branch_layers.append(Transpose())

                reg_branch_layers.append(Transpose())
                reg_branch_layers.append(nn.LayerNorm(inplanes))
                reg_branch_layers.append(Transpose())

            cls_branch_layers.append(
                nn.Conv1d(inplanes,
                          inplanes,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            # cls_branch_layers.append(nn.BatchNorm1d(inplanes))
            if dout_p > 0:
                cls_branch_layers.append(nn.Dropout(dout_p))
            cls_branch_layers.append(nn.ReLU(inplace=True))

            reg_branch_layers.append(nn.Conv1d(inplanes,
                                               inplanes,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1))
            # reg_branch_layers.append(nn.BatchNorm1d(inplanes))
            if dout_p > 0:
                reg_branch_layers.append(nn.Dropout(dout_p))
            reg_branch_layers.append(nn.ReLU(inplace=True))

        self.reg_head = nn.Sequential(*reg_branch_layers)
        self.cls_head = nn.Sequential(*cls_branch_layers)

        self.reg_out = nn.Conv1d(inplanes,
                                 2,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.confidence_out = nn.Conv1d(inplanes,
                                        1,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        self.center_out = nn.Conv1d(inplanes,
                                    1,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)

        # self.relu = nn.ReLU(inplace=True)

        self.scale_exp = nn.Parameter(torch.FloatTensor([1., 1., 1., 1., 1., 1., 1., 1.]), requires_grad=True)

        prior = prior
        b = -math.log((1 - prior) / prior)
        self.confidence_out.bias.data.fill_(b)

        # model parameter init
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

    # def forward(self, x, scale):
    def forward(self, x_fpn):
        reg_heads = []
        center_heads = []
        confidence_heads = []

        # for out, scale in zip(x_fpn,self.scales):
        for i, x in enumerate(x_fpn):
            reg_head_out = self.reg_head(x)
            cls_head_out = self.cls_head(x)

            # (B, 1/2, Sm)-->(B, Sm, 1/2)
            reg_heads.append(self.reg_out(reg_head_out).permute(0, 2, 1) * self.scale_exp[i])
            center_heads.append(self.center_out(reg_head_out).permute(0, 2, 1))
            confidence_heads.append(self.confidence_out(cls_head_out).permute(0, 2, 1))

        return reg_heads, center_heads, confidence_heads


class FCOSPositions(nn.Module):
    def __init__(self, cfg, strides_fcos):
        super(FCOSPositions, self).__init__()
        self.strides_fcos = strides_fcos
        self.cfg = cfg

    def forward(self, batch_size, fpn_feature_sizes):
        """
        generate batch positions
        """
        fpn_feature_sizes = torch.tensor(fpn_feature_sizes).float().to(self.cfg.device)
        device = fpn_feature_sizes.device

        one_sample_positions = []
        strides = torch.tensor(self.strides_fcos, device=device)

        for stride, fpn_feature_size in zip(strides, fpn_feature_sizes):
            featrue_positions = self.generate_positions_on_feature_map(
                fpn_feature_size, stride)
            featrue_positions = featrue_positions.to(device)
            one_sample_positions.append(featrue_positions)

        batch_positions = []
        for per_level_featrue_positions in one_sample_positions:
            per_level_featrue_positions = per_level_featrue_positions.unsqueeze(
                0).repeat(batch_size, 1)                  #对于每一个样本来说，其位置映射均是一样的，只要输入形状和网络结构不变
            batch_positions.append(per_level_featrue_positions)

        # if input size:[B,1024,2400]
        # batch_positions shape:[[B, 600],[B, 300],[B, 150],[B, 75],[B, 38]]
        # per position format:[l_center]
        return batch_positions

    def generate_positions_on_feature_map(self, feature_map_size, stride):
        """
        generate all positions on a feature map
        """
        feature_map_positions = ((torch.arange(0, feature_map_size, device=stride.device) + 0.5) * stride)

        return feature_map_positions


def compute_one_video_giou_loss(per_image_reg_preds, per_image_targets, obj_mask):
    """
    compute one image giou loss(reg loss)
    per_image_reg_preds:[points_num,２]
    per_image_targets:[anchor_num,4]
    positive_index:[positive_num,]
    """
    # only use positive points sample to compute reg loss
    # 按照类别进行
    # print(obj_mask.nonzero())

    device = per_image_reg_preds.device
    per_image_reg_pred = per_image_reg_preds[obj_mask]
    per_image_target = per_image_targets[obj_mask].to(device)
    positive_points_num = per_image_reg_pred.shape[0]
    # print('per_image_reg_pred:\n', per_image_reg_pred)
    # print('per_image_target:\n', per_image_target)

    if positive_points_num == 0:
        return torch.tensor(0.).to(device)

    center_ness_targets = per_image_target[:, 2]

    pred_bboxes_xy_min = (per_image_target[:, 3] - per_image_reg_pred[:, 0]).unsqueeze(
        1)
    pred_bboxes_xy_max = (per_image_target[:, 3] + per_image_reg_pred[:, 1]).unsqueeze(
        1)

    gt_bboxes_xy_min = (per_image_target[:, 3] - per_image_target[:, 0]).unsqueeze(1)
    gt_bboxes_xy_max = (per_image_target[:, 3] + per_image_target[:, 1]).unsqueeze(1)

    # 求得在grid中的start&end
    pred_bboxes = torch.cat([pred_bboxes_xy_min, pred_bboxes_xy_max], dim=-1)
    gt_bboxes = torch.cat([gt_bboxes_xy_min, gt_bboxes_xy_max], dim=-1)
    # -----------------print('gt_bboxes:\n', gt_bboxes)

    overlap_length_left = torch.max(pred_bboxes[:, 0], gt_bboxes[:, 0])
    overlap_length_right = torch.min(pred_bboxes[:, 1], gt_bboxes[:, 1])

    # 求两者的交集
    overlap_length = torch.clamp(overlap_length_right - overlap_length_left, min=0)
    # print(overlap_length)

    # anchors and annotations convert format to [x1,y1,w,h]
    pred_bboxes_l = pred_bboxes[:, 1] - pred_bboxes[:, 0] + 1
    gt_bboxes_l = gt_bboxes[:, 1] - gt_bboxes[:, 0] + 1

    # compute union_area,求两者的并集
    union_length = pred_bboxes_l + gt_bboxes_l - overlap_length
    union_length = torch.clamp(union_length, min=1e-4)
    # compute ious between one image anchors and one image annotations
    ious = overlap_length / union_length

    enclose_length_left = torch.min(pred_bboxes[:, 0], gt_bboxes[:, 0])
    enclose_length_right = torch.max(pred_bboxes[:, 1], gt_bboxes[:, 1])
    enclose_length = torch.clamp(enclose_length_right - enclose_length_left, min=0)
    enclose_length = torch.clamp(enclose_length, min=1e-4)

    # giou = ious - (enclose_length - union_length) / enclose_length
    # 对于我们的时序数据来说，giou=iou，其值越大越好，对应的loss越小越好
    gious_loss = 1. - ious + (enclose_length - union_length) / enclose_length
    gious_loss = torch.clamp(gious_loss, min=-1.0, max=1.0)
    # use center_ness_targets as the weight of gious loss
    gious_loss = gious_loss * center_ness_targets
    gious_loss = gious_loss.sum() / positive_points_num
    gious_loss = 2. * gious_loss  # 最后乘以2是为了平衡回归loss与其他loss的数量级。
    #print('gious_loss:\n', gious_loss)

    return gious_loss


def compute_one_video_center_ness_loss(per_video_center_preds,
                                       per_video_targets, obj_mask):
    """
    compute one image center_ness loss(center ness loss)
    per_image_center_preds:[points_num,3]
    per_image_targets:[anchor_num,4]
    """
    # only use positive points sample to compute center_ness loss
    device = per_video_center_preds.device
    per_video_center_pred = per_video_center_preds[obj_mask]
    per_video_target = per_video_targets[obj_mask]
    positive_points_num = per_video_center_pred.shape[0]
    # print('per_video_center_pred:\n', per_video_center_pred)
    # print('per_image_targets_center:\n', per_video_target[:, 2])

    if positive_points_num == 0:
        return torch.tensor(0.).to(device)

    center_ness_target = per_video_target[:, 2].to(per_video_center_pred.device)
    # print('center_ness_targets:\n', center_ness_targets[positive_index])
    # print('per_image_center_preds[:,2]:\n', per_video_center_preds[positive_index, 2])

    center_ness_loss = -(
            center_ness_target[:] * torch.log(per_video_center_pred[:]) +
            (1. - center_ness_target[:]) *
            torch.log(1. - per_video_center_pred[:]))
    # print('center_ness_loss:\n', center_ness_loss)
    center_ness_loss = center_ness_loss.sum() / positive_points_num
    # ---------------------print('center_ness_loss:\n', center_ness_loss)
    #print('centerness_loss:\n', center_ness_loss)

    return center_ness_loss


class ProposalGeneratorFCOS(nn.Module):

    def __init__(self, cfg):
        super(ProposalGeneratorFCOS, self).__init__()

        self.cfg = cfg
        self.dout_p = cfg.dout_p_fcos
        self.layer_norm = cfg.layer_norm
        self.planes = cfg.planes
        self.C2_inplanes = cfg.C2_inplanes
        self.C3_inplanes = cfg.C3_inplanes
        self.C4_inplanes = cfg.C4_inplanes
        self.C5_inplanes = cfg.C5_inplanes

        # modality choose
        if cfg.modality == 'audio':
            self.d_model_modality = cfg.d_model_audio
            self.d_raw_modality = cfg.d_aud
        elif cfg.modality == 'video':
            self.d_model_modality = cfg.d_model_video
            self.d_raw_modality = cfg.d_vid
        elif cfg.modality == 'text':
            self.d_model_modality = cfg.d_model_text
            self.d_raw_modality = cfg.d_text
        else:
            raise NotImplementedError

        self.norm = nn.LayerNorm(self.d_raw_modality)
        # self.norm = nn.BatchNorm1d(cfg.text_s)

        self.linear = nn.Linear(self.d_raw_modality, cfg.C2_inplanes)
        # self.conv1d = nn.Conv1d(self.d_raw_modality, cfg.C2_inplanes,
        #                         kernel_size=3, stride=1, padding=1)

        # Positions
        self.positions = FCOSPositions(cfg, cfg.strides_fcos)

        # Backbone network
        self.backbone = BackBone(self.C2_inplanes, self.C3_inplanes, self.C4_inplanes, self.C5_inplanes)

        # FPN network
        self.fpn = FPN(self.C3_inplanes, self.C4_inplanes, self.C5_inplanes, self.planes, use_p5=True)

        # Heads network
        self.reg_cen_cls_head = FCOSRegCenterClsHead(self.layer_norm, self.planes, self.dout_p)

        # self.scale_exp = nn.ModuleList([ScaleExp(1.0) for _ in range(8)])

        self.bce_loss = nn.BCELoss()
        # self.mse_loss = nn.MSELoss()

    def forward(self, x, targets):
        '''
            targets:(*, 4), 4: batch_id, s, e, proposal_id
            x:(B, S, d_a/d_v/d_t)
        '''
        B, _, _ = x.shape
        # x = self.norm(x).permute(0, 2, 1)
        # (B, S, d_modility)-->(B, S, d_model_fcos)
        x = self.linear(x)
    # with torchsnooper.snoop():
        timespan_fcos = torch.tensor(self.cfg.timespan_fcos, device=x.device)
        # x = self.conv1d(x)

        # x(B, S, d_model_fcos)-->(B, d_model_fcos, S)
        x = x.permute(0, 2, 1)
        x = self.backbone(x)
        x = self.fpn(x)
        reg_heads, center_heads, confidence_heads = self.reg_cen_cls_head(x)
        batch_positions = self.positions(B, self.cfg.fpn_feature_sizes)

        # 将reg_pred、center_preds、all_points_position、all_points_mi按照dim=1维度进行拼接，把所有层的信息按顺序放在一个张量中
        # [B, 2392, 2]
        reg_preds_tensor = torch.cat(reg_heads, dim=1)
        reg_preds_tensor = torch.exp(reg_preds_tensor)
        # [B, 2392, 1]
        center_preds_tensor = torch.cat(center_heads, dim=1)
        center_preds_tensor = torch.sigmoid(center_preds_tensor)
        # [B, 2392, 2]
        confidence_preds_tensor = torch.cat(confidence_heads, dim=1)
        confidence_preds_tensor = torch.sigmoid(confidence_preds_tensor)
        # [B, 2392]-->(B,2392,1)
        # batch_positions:list, (B, layer_points_num)-->(B, all_points_num,1)
        all_points_position = torch.cat(batch_positions, dim=1).unsqueeze(dim=-1).to(center_preds_tensor.device)
        # predictions:(B, 2392, 4), 4: l,r,center,position
        # print('debug:\n', reg_preds_tensor.shape, center_preds_tensor.shape, all_points_position.shape)
        preds = torch.cat((reg_preds_tensor, center_preds_tensor, all_points_position, confidence_preds_tensor), dim=-1)

        # we need to detach them from the graph as we don't need to back_proparate
        # on them
        predictions = preds.clone().detach().to(self.cfg.device)

        # from grid-axis to second
        predictions[:, :, 0:2] = predictions[:, :, 0:2] * (timespan_fcos.to(self.cfg.device))
        predictions[:, :, 3] = predictions[:, :, 3] * (timespan_fcos.to(self.cfg.device))

        batch_av_loss = 0
        losses_dict = {}
        if targets is not None:
            # batch_targets:(B, points_num, 4), batch_positive:(B,)
            batch_targets, objs_mask, noobjs_mask, targets_obj = get_batch_position_annotations(
                               self.cfg, preds, reg_heads, all_points_position, targets, self.cfg.timespan_fcos)
            batch_targets[:, :, 2] = torch.sigmoid(batch_targets[:, :, 2])

            reg_loss = 0
            centerness_loss = 0
            loss_conf = 0
            for prediction, trg, obj_mask, noobj_mask, target_obj in zip(preds, batch_targets, objs_mask, noobjs_mask, targets_obj):
                # print('centerness_trg:\n', trg[obj_mask])
                print('--------------------开始验证！-------------------')
                ## Loss
                # reg loss
                l_r = prediction[:, 0:2]
                # GIOU Loss
                reg_loss += compute_one_video_giou_loss(l_r, trg, obj_mask)

                # centerness loss
                pred_center = prediction[:, 2]
                # print('c:\n', pred_center[obj_mask])
                targets_center = trg[:, 2]
                # print('targets_center:\n', targets_center[obj_mask])
                # print('no_targets_center:\n', targets_center[noobj_mask])
                # centerness_loss += compute_one_video_center_ness_loss(c, trg, obj_mask)
                centerness_loss += self.bce_loss(pred_center[obj_mask], targets_center[obj_mask])

                # confidence loss
                score = prediction[:, 4]
                # print('score_obj:\n', score[obj_mask])
                # print('target_obj:\n', target_obj[obj_mask])
                # print('score_noobj:\n', score[noobj_mask])
                # print('target_noobj:\n', target_obj[noobj_mask])
                loss_obj = self.bce_loss(score[obj_mask], target_obj[obj_mask])
                # print('loss_obj:\n', loss_obj)
                loss_noobj = self.bce_loss(score[noobj_mask], target_obj[noobj_mask])
                # print('loss_noobj:\n', loss_noobj)
                loss_conf += self.cfg.obj_coeff * loss_obj + self.cfg.noobj_coeff * loss_noobj

                # Total Loss
            batch_av_loss = (self.cfg.reg_coeff * reg_loss + self.cfg.cen_coeff * centerness_loss + loss_conf) / B

            #print('batch_loss:\n', reg_loss / B, centerness_loss / B, loss_conf / B )
            losses_dict = {
                'reg_loss': reg_loss / B,
                'centerness_loss': centerness_loss / B,
                'loss_conf': loss_conf / B}

        return predictions, batch_av_loss, losses_dict


class TrimodalProposalGeneratorFCOSNoEncoder(nn.Module):

    def __init__(self, cfg):
        super(TrimodalProposalGeneratorFCOSNoEncoder, self).__init__()
        # assert cfg.modality == 'audio_video_text'
        self.cfg = cfg
        self.dout_p = cfg.dout_p_fcos
        print('self.dout_p_fcos:\n', self.dout_p)
        self.layer_norm = cfg.layer_norm
        self.planes = cfg.planes
        self.C2_inplanes = cfg.C2_inplanes
        self.C3_inplanes = cfg.C3_inplanes
        self.C4_inplanes = cfg.C4_inplanes
        self.C5_inplanes = cfg.C5_inplanes

        if cfg.feature_fusion_mode == 'add':
            self.d_raw_modality = cfg.d_model_add     # 1024
        elif cfg.feature_fusion_mode == 'cat':
            self.d_raw_modality = cfg.d_model_cat     # 2048
        self.linear_V = nn.Linear(cfg.d_model_video, self.d_raw_modality)
        self.linear_A = nn.Linear(cfg.d_model_audio, self.d_raw_modality)
        self.linear_T = nn.Linear(cfg.d_model_text, self.d_raw_modality)

        self.linear_F = nn.Linear(self.d_raw_modality, self.C2_inplanes)

        if cfg.use_linear_embedder:
            self.emb_V = FeatureEmbedder(cfg.d_vid, cfg.d_model_video)
            self.emb_A = FeatureEmbedder(cfg.d_aud, cfg.d_model_audio)
            self.emb_T = FeatureEmbedder(cfg.d_text, cfg.d_model_text)
        else:
            self.emb_V = Identity()
            self.emb_A = Identity()
            self.emb_T = Identity()
        self.pos_enc_V = PositionalEncoder(cfg.d_model_video, self.dout_p)  # 300
        self.pos_enc_A = PositionalEncoder(cfg.d_model_audio, self.dout_p)  # 800
        self.pos_enc_T = PositionalEncoder(cfg.d_model_text, self.dout_p)  # 2400


        # Positions
        self.positions = FCOSPositions(cfg, cfg.strides_fcos)

        # Backbone network
        self.backbone = BackBone(self.C2_inplanes, self.C3_inplanes, self.C4_inplanes, self.C5_inplanes)

        # FPN network
        self.fpn = FPN(self.C3_inplanes, self.C4_inplanes, self.C5_inplanes, self.planes, use_p5=True)

        # Heads network
        self.reg_cen_cls_head = FCOSRegCenterClsHead(self.layer_norm, self.planes, self.dout_p)

        self.bce_loss = nn.BCELoss()

    def forward(self, x, targets, masks):
        """
        :param x: (video(B, S, Dv), audio(B, S, Da), text(B, S, Dt))
        :param masks: {'V_mask':(B, 1, S), 'A_mask':(B, 1, S), 'T_mask':(B, 1, S), 'AV_mask':(B, 1, S)}
        :return: predictions, batch_av_loss, losses_dict
        """
        A, V, T = x
        print(A.shape, V.shape, T.shape)

        # １、特征预处理
        # (B, Sm, Dm)-->(B, Sm, Dm), m in (v,a,t)
        A = self.emb_A(A)
        V = self.emb_V(V)
        T = self.emb_T(T)
        A = self.pos_enc_A(A)
        V = self.pos_enc_V(V)
        T = self.pos_enc_T(T)

        # ２、特征融合编码
        # x_encode:(B, S, D_fcos)
        # x_encode, _ = self.encoder((A, V, T), masks, None)
        V = self.linear_V(V)
        A = self.linear_A(A)
        T = self.linear_T(T)
        AV = torch.add(A,V)
        AVT = torch.add(AV,T)
        # x_encode = self.linear(x_encode)
        # (B, S, D)-->(B, D, S)
        # x_encode = x_encode.permute(0, 2, 1)
        # (B, D, S)-->(B, d, S)
        # x_encode = self.conv1d_1(x_encode)
        # x_encode = self.conv1d_2(x_encode)
        x_encode = self.linear_F(AVT)
        # x_encode = self.linear_2(x_encode)

        # 3、fcos实现proposal训练
        # x_encode:(B, 2400, 2048)
        B, _, _ = x_encode.shape

        # x(B, S, d_model_fcos)-->(B, d_model_fcos, S)
        x_encode = x_encode.permute(0, 2, 1)
        x_encode = self.backbone(x_encode)
        x_encode = self.fpn(x_encode)
        reg_heads, center_heads, confidence_heads = self.reg_cen_cls_head(x_encode)
        batch_positions = self.positions(B, self.cfg.fpn_feature_sizes)

        # 将reg_pred、center_preds、all_points_position、all_points_mi按照dim=1维度进行拼接，把所有层的信息按顺序放在一个张量中
        # [B, 2392, 2]
        reg_preds_tensor = torch.cat(reg_heads, dim=1)
        reg_preds_tensor = torch.exp(reg_preds_tensor)
        # [B, 2392, 1]
        center_preds_tensor = torch.cat(center_heads, dim=1)
        center_preds_tensor = torch.sigmoid(center_preds_tensor)
        # [B, 2392, 2]
        confidence_preds_tensor = torch.cat(confidence_heads, dim=1)
        confidence_preds_tensor = torch.sigmoid(confidence_preds_tensor)
        # [B, 2392]-->(B,2392,1)
        # batch_positions:list, (B, layer_points_num)-->(B, all_points_num,1)
        all_points_position = torch.cat(batch_positions, dim=1).unsqueeze(dim=-1).to(self.cfg.device)
        # predictions:(B, 2392, 4), 4: l,r,center,position
        # print('debug:\n', reg_preds_tensor.shape, center_preds_tensor.shape, all_points_position.shape)
        preds = torch.cat((reg_preds_tensor, center_preds_tensor, all_points_position, confidence_preds_tensor), dim=-1)

        # we need to detach them from the graph as we don't need to back_proparate
        # on them
        predictions = preds.clone().detach()

        # from grid-axis to second
        predictions[:, :, 0:2] = predictions[:, :, 0:2] * self.cfg.timespan_fcos
        predictions[:, :, 3] = predictions[:, :, 3] * self.cfg.timespan_fcos

        batch_av_loss = 0
        losses_dict = {}
        if targets is not None:
            # batch_targets:(B, points_num, 4), batch_positive:(B,)
            batch_targets, objs_mask, noobjs_mask, targets_obj = get_batch_position_annotations(
                self.cfg, preds, reg_heads, all_points_position, targets, self.cfg.timespan_fcos)
            batch_targets[:, :, 2] = torch.sigmoid(batch_targets[:, :, 2])

            reg_loss = torch.tensor([0.], device=preds.device)
            centerness_loss = torch.tensor([0.], device=preds.device)
            loss_conf = torch.tensor([0.], device=preds.device)
            for prediction, trg, obj_mask, noobj_mask, target_obj in zip(preds, batch_targets, objs_mask, noobjs_mask,
                                                                         targets_obj):
                trg = trg.to(prediction.device)
                # ndarray, (*,),
                # positive = torch.from_numpy(positive)
                print('--------------------开始验证！-------------------')
                ## Loss
                # reg loss
                l_r = prediction[:, 0:2]
                reg_loss += compute_one_video_giou_loss(l_r, trg, obj_mask)

                # centerness loss
                pred_center = prediction[:, 2]
                # print('c:\n', pred_center[obj_mask])
                targets_center = trg[:, 2]

                centerness_loss += self.bce_loss(pred_center[obj_mask], targets_center[obj_mask])

                # confidence loss
                score = prediction[:, 4]
                loss_obj = self.bce_loss(score[obj_mask], target_obj[obj_mask])
                print('loss_obj:\n', loss_obj)
                loss_noobj = self.bce_loss(score[noobj_mask], target_obj[noobj_mask])
                print('loss_noobj:\n', loss_noobj)
                loss_conf += self.cfg.obj_coeff * loss_obj + self.cfg.noobj_coeff * loss_noobj

                # Total Loss
            batch_av_loss = (self.cfg.reg_coeff * reg_loss + self.cfg.cen_coeff * centerness_loss + loss_conf) / B

            print('batch_loss:\n', reg_loss / B, centerness_loss / B, loss_conf / B)
            losses_dict = {
                'reg_loss': reg_loss / B,
                'centerness_loss': centerness_loss / B,
                'loss_conf': loss_conf / B}

        return predictions, batch_av_loss, losses_dict


class BimodalProposalGeneratorFCOSNoEncoder(nn.Module):

    def __init__(self, cfg):
        super(BimodalProposalGeneratorFCOSNoEncoder, self).__init__()
        # assert cfg.modality == 'audio_video_text'
        self.cfg = cfg
        self.dout_p = cfg.dout_p_fcos
        print('self.dout_p_fcos:\n', self.dout_p)
        self.layer_norm = cfg.layer_norm
        self.planes = cfg.planes
        self.C2_inplanes = cfg.C2_inplanes
        self.C3_inplanes = cfg.C3_inplanes
        self.C4_inplanes = cfg.C4_inplanes
        self.C5_inplanes = cfg.C5_inplanes

        if cfg.feature_fusion_mode == 'add':
            self.d_raw_modality = cfg.d_model_add     # 1024
        elif cfg.feature_fusion_mode == 'cat':
            self.d_raw_modality = cfg.d_model_cat     # 2048
        self.linear_V = nn.Linear(cfg.d_model_video, self.d_raw_modality)
        self.linear_A = nn.Linear(cfg.d_model_audio, self.d_raw_modality)
        # self.linear_T = nn.Linear(cfg.d_model_text, self.d_raw_modality)

        self.linear_F = nn.Linear(self.d_raw_modality, self.C2_inplanes)

        if cfg.use_linear_embedder:
            self.emb_V = FeatureEmbedder(cfg.d_vid, cfg.d_model_video)
            self.emb_A = FeatureEmbedder(cfg.d_aud, cfg.d_model_audio)
            # self.emb_T = FeatureEmbedder(cfg.d_text, cfg.d_model_text)
        else:
            self.emb_V = Identity()
            self.emb_A = Identity()
            # self.emb_T = Identity()
        self.pos_enc_V = PositionalEncoder(cfg.d_model_video, self.dout_p)  # 300
        self.pos_enc_A = PositionalEncoder(cfg.d_model_audio, self.dout_p)  # 800
        # self.pos_enc_T = PositionalEncoder(cfg.d_model_text, self.dout_p)  # 2400


        # Positions
        self.positions = FCOSPositions(cfg, cfg.strides_fcos)

        # Backbone network
        self.backbone = BackBone(self.C2_inplanes, self.C3_inplanes, self.C4_inplanes, self.C5_inplanes)

        # FPN network
        self.fpn = FPN(self.C3_inplanes, self.C4_inplanes, self.C5_inplanes, self.planes, use_p5=True)

        # Heads network
        self.reg_cen_cls_head = FCOSRegCenterClsHead(self.layer_norm, self.planes, self.dout_p)

        self.bce_loss = nn.BCELoss()

    def forward(self, x, targets, masks):
        """
        :param x: (video(B, S, Dv), audio(B, S, Da), text(B, S, Dt))
        :param masks: {'V_mask':(B, 1, S), 'A_mask':(B, 1, S), 'T_mask':(B, 1, S), 'AV_mask':(B, 1, S)}
        :return: predictions, batch_av_loss, losses_dict
        """
        A, V, T = x
        print(A.shape, V.shape)

        # １、特征预处理
        # (B, Sm, Dm)-->(B, Sm, Dm), m in (v,a,t)
        A = self.emb_A(A)
        V = self.emb_V(V)
        # T = self.emb_T(T)
        A = self.pos_enc_A(A)
        V = self.pos_enc_V(V)
        # T = self.pos_enc_T(T)

        # ２、特征融合编码
        # x_encode:(B, S, D_fcos)
        # x_encode, _ = self.encoder((A, V, T), masks, None)
        V = self.linear_V(V)
        A = self.linear_A(A)
        # T = self.linear_T(T)
        AV = torch.add(A,V)
        # AVT = torch.add(AV,T)
        # x_encode = self.linear(x_encode)
        # (B, S, D)-->(B, D, S)
        # x_encode = x_encode.permute(0, 2, 1)
        # (B, D, S)-->(B, d, S)
        # x_encode = self.conv1d_1(x_encode)
        # x_encode = self.conv1d_2(x_encode)
        x_encode = self.linear_F(AV)
        # x_encode = self.linear_2(x_encode)

        # 3、fcos实现proposal训练
        # x_encode:(B, 2400, 2048)
        B, _, _ = x_encode.shape

        # x(B, S, d_model_fcos)-->(B, d_model_fcos, S)
        x_encode = x_encode.permute(0, 2, 1)
        x_encode = self.backbone(x_encode)
        x_encode = self.fpn(x_encode)
        reg_heads, center_heads, confidence_heads = self.reg_cen_cls_head(x_encode)
        batch_positions = self.positions(B, self.cfg.fpn_feature_sizes)

        # 将reg_pred、center_preds、all_points_position、all_points_mi按照dim=1维度进行拼接，把所有层的信息按顺序放在一个张量中
        # [B, 2392, 2]
        reg_preds_tensor = torch.cat(reg_heads, dim=1)
        reg_preds_tensor = torch.exp(reg_preds_tensor)
        # [B, 2392, 1]
        center_preds_tensor = torch.cat(center_heads, dim=1)
        center_preds_tensor = torch.sigmoid(center_preds_tensor)
        # [B, 2392, 2]
        confidence_preds_tensor = torch.cat(confidence_heads, dim=1)
        confidence_preds_tensor = torch.sigmoid(confidence_preds_tensor)
        # [B, 2392]-->(B,2392,1)
        # batch_positions:list, (B, layer_points_num)-->(B, all_points_num,1)
        all_points_position = torch.cat(batch_positions, dim=1).unsqueeze(dim=-1).to(self.cfg.device)
        # predictions:(B, 2392, 4), 4: l,r,center,position
        # print('debug:\n', reg_preds_tensor.shape, center_preds_tensor.shape, all_points_position.shape)
        preds = torch.cat((reg_preds_tensor, center_preds_tensor, all_points_position, confidence_preds_tensor), dim=-1)

        # we need to detach them from the graph as we don't need to back_proparate
        # on them
        predictions = preds.clone().detach()

        # from grid-axis to second
        predictions[:, :, 0:2] = predictions[:, :, 0:2] * self.cfg.timespan_fcos
        predictions[:, :, 3] = predictions[:, :, 3] * self.cfg.timespan_fcos

        batch_av_loss = 0
        losses_dict = {}
        if targets is not None:
            # batch_targets:(B, points_num, 4), batch_positive:(B,)
            batch_targets, objs_mask, noobjs_mask, targets_obj = get_batch_position_annotations(
                self.cfg, preds, reg_heads, all_points_position, targets, self.cfg.timespan_fcos)
            batch_targets[:, :, 2] = torch.sigmoid(batch_targets[:, :, 2])

            reg_loss = torch.tensor([0.], device=preds.device)
            centerness_loss = torch.tensor([0.], device=preds.device)
            loss_conf = torch.tensor([0.], device=preds.device)
            for prediction, trg, obj_mask, noobj_mask, target_obj in zip(preds, batch_targets, objs_mask, noobjs_mask,
                                                                         targets_obj):
                trg = trg.to(prediction.device)
                # ndarray, (*,),
                # positive = torch.from_numpy(positive)
                print('--------------------开始验证！-------------------')
                ## Loss
                # reg loss
                l_r = prediction[:, 0:2]
                reg_loss += compute_one_video_giou_loss(l_r, trg, obj_mask)

                # centerness loss
                pred_center = prediction[:, 2]
                # print('c:\n', pred_center[obj_mask])
                targets_center = trg[:, 2]

                centerness_loss += self.bce_loss(pred_center[obj_mask], targets_center[obj_mask])

                # confidence loss
                score = prediction[:, 4]
                loss_obj = self.bce_loss(score[obj_mask], target_obj[obj_mask])
                print('loss_obj:\n', loss_obj)
                loss_noobj = self.bce_loss(score[noobj_mask], target_obj[noobj_mask])
                print('loss_noobj:\n', loss_noobj)
                loss_conf += self.cfg.obj_coeff * loss_obj + self.cfg.noobj_coeff * loss_noobj

                # Total Loss
            batch_av_loss = (self.cfg.reg_coeff * reg_loss + self.cfg.cen_coeff * centerness_loss + loss_conf) / B

            print('batch_loss:\n', reg_loss / B, centerness_loss / B, loss_conf / B)
            losses_dict = {
                'reg_loss': reg_loss / B,
                'centerness_loss': centerness_loss / B,
                'loss_conf': loss_conf / B}

        return predictions, batch_av_loss, losses_dict


class TrimodalProposalGeneratorFCOS(nn.Module):

    def __init__(self, cfg):
        super(TrimodalProposalGeneratorFCOS, self).__init__()
        assert cfg.modality == 'audio_video_text'
        self.cfg = cfg
        self.dout_p = cfg.dout_p_fcos
        self.layer_norm = cfg.layer_norm
        self.planes = cfg.planes
        self.C3_inplanes = cfg.C3_inplanes
        self.C4_inplanes = cfg.C4_inplanes
        self.C5_inplanes = cfg.C5_inplanes

        if cfg.AV_fusion_mode == 'add':
            self.d_model_mid = cfg.d_model//4
        else:
            self.d_model_mid = cfg.d_model//2

        if cfg.AVT_fusion_mode == 'add':
            self.d_raw_fcos = cfg.d_model//4
        else:
            self.d_raw_fcos = cfg.d_model//2

        self.emb_V = Identity()
        self.emb_A = Identity()
        self.emb_T = Identity()
        self.pos_enc_V = PositionalEncoder(cfg.d_model_video, self.dout_p)  # 300
        self.pos_enc_A = PositionalEncoder(cfg.d_model_audio, self.dout_p)  # 800
        self.pos_enc_T = PositionalEncoder(cfg.d_model_text, self.dout_p)   # 2400

        if cfg.pretrained_cap_model_path is not None:
            print(f'Pretrained caption path: \n {cfg.pretrained_cap_model_path}')
            cap_model_cpt = torch.load(cfg.pretrained_cap_model_path, map_location='cpu')
            pre_cfg = cap_model_cpt['config']
            self.encoder = TriModalEncoder(pre_cfg, pre_cfg.d_model_audio, pre_cfg.d_model_video,
                                           pre_cfg.d_model_text, pre_cfg.d_model, pre_cfg.dout_p,
                                           pre_cfg.d_ff_audio, pre_cfg.d_ff_video, pre_cfg.d_ff_text)
            encoder_weights = {k: v for k, v in cap_model_cpt['model_state_dict'].items() if 'encoder' in k}
            encoder_weights = {k.replace('module.encoder.', ''): v for k, v in encoder_weights.items()}
            self.encoder.load_state_dict(encoder_weights)
            self.encoder = self.encoder.to(cfg.device)
            for param in self.encoder.parameters():
                param.requires_grad = cfg.finetune_cap_encoder
        else:
            self.encoder = TriModalEncoder(cfg, cfg.d_model_audio, cfg.d_model_video, cfg.d_model_text, cfg.d_model,
                                           cfg.dout_p, cfg.d_ff_audio, cfg.d_ff_video, cfg.d_ff_text)
            for p in self.encoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        # Positions
        self.positions = FCOSPositions(cfg, cfg.strides_fcos)

        # Backbone network
        self.backbone_Av = BackBone(self.d_model_mid, self.C3_inplanes, self.C4_inplanes, self.C5_inplanes)
        self.backbone_Va = BackBone(self.d_model_mid, self.C3_inplanes, self.C4_inplanes, self.C5_inplanes)
        self.backbone_AVT = BackBone(self.d_raw_fcos, self.C3_inplanes, self.C4_inplanes, self.C5_inplanes)

        # FPN network
        self.fpn_Av = FPN(self.C3_inplanes, self.C4_inplanes, self.C5_inplanes, self.planes, use_p5=True)
        self.fpn_Va = FPN(self.C3_inplanes, self.C4_inplanes, self.C5_inplanes, self.planes, use_p5=True)
        self.fpn_AVT = FPN(self.C3_inplanes, self.C4_inplanes, self.C5_inplanes, self.planes, use_p5=True)

        # Heads network
        self.head_Av = FCOSRegCenterClsHead(self.layer_norm, self.planes, self.dout_p)
        self.head_Va = FCOSRegCenterClsHead(self.layer_norm, self.planes, self.dout_p)
        self.head_AVT = FCOSRegCenterClsHead(self.layer_norm, self.planes, self.dout_p)

        self.bce_loss = nn.BCELoss()

    def fcos_prop(self, backbone, fpn, head, enc, tar):
        B, _, _ = enc.shape
        enc = enc.permute(0, 2, 1)
        enc = backbone(enc)
        enc = fpn(enc)
        reg_heads, center_heads, confidence_heads = head(enc)
        batch_positions = self.positions(B, self.cfg.fpn_feature_sizes)

        # 将reg_pred、center_preds、all_points_position、all_points_mi按照dim=1维度进行拼接，把所有层的信息按顺序放在一个张量中
        # [B, 2392, 2]
        reg_preds_tensor = torch.cat(reg_heads, dim=1)
        reg_preds_tensor = torch.exp(reg_preds_tensor)
        # [B, 2392, 1]
        center_preds_tensor = torch.cat(center_heads, dim=1)
        center_preds_tensor = torch.sigmoid(center_preds_tensor)
        # [B, 2392, 2]
        confidence_preds_tensor = torch.cat(confidence_heads, dim=1)
        confidence_preds_tensor = torch.sigmoid(confidence_preds_tensor)
        # [B, 2392]-->(B,2392,1)
        # batch_positions:list, (B, layer_points_num)-->(B, all_points_num,1)
        all_points_position = torch.cat(batch_positions, dim=1).unsqueeze(dim=-1).to(self.cfg.device)
        # predictions:(B, 2392, 4), 4: l,r,center,position
        preds = torch.cat((reg_preds_tensor, center_preds_tensor, all_points_position, confidence_preds_tensor), dim=-1)

        # we need to detach them from the graph as we don't need to back_proparate on them
        predictions = preds.clone().detach()

        # from grid-axis to second
        predictions[:, :, 0:2] = predictions[:, :, 0:2] * self.cfg.timespan_fcos
        predictions[:, :, 3] = predictions[:, :, 3] * self.cfg.timespan_fcos

        batch_av_loss = 0
        losses_dict = {}
        if tar is not None:
            # batch_targets:(B, points_num, 4), batch_positive:(B,)
            batch_targets, objs_mask, noobjs_mask, targets_obj = get_batch_position_annotations(
                self.cfg, preds, reg_heads, all_points_position, tar, self.cfg.timespan_fcos)
            batch_targets[:, :, 2] = torch.sigmoid(batch_targets[:, :, 2])

            reg_loss = torch.tensor([0.], device=preds.device)
            centerness_loss = torch.tensor([0.], device=preds.device)
            loss_conf = torch.tensor([0.], device=preds.device)
            for prediction, trg, obj_mask, noobj_mask, target_obj in zip(preds, batch_targets, objs_mask, noobjs_mask, targets_obj):
                trg = trg.to(prediction.device)
                l_r = prediction[:, 0:2]
                reg_loss += compute_one_video_giou_loss(l_r, trg, obj_mask)

                # centerness loss
                pred_center = prediction[:, 2]
                targets_center = trg[:, 2]
                centerness_loss += self.bce_loss(pred_center[obj_mask], targets_center[obj_mask])

                # confidence loss
                score = prediction[:, 4]
                loss_obj = self.bce_loss(score[obj_mask], target_obj[obj_mask])
                loss_noobj = self.bce_loss(score[noobj_mask], target_obj[noobj_mask])
                loss_conf += self.cfg.obj_coeff * loss_obj + self.cfg.noobj_coeff * loss_noobj

            batch_av_loss = (self.cfg.reg_coeff * reg_loss + self.cfg.cen_coeff * centerness_loss + loss_conf) / B

            losses_dict = {
                'reg_loss': reg_loss / B,
                'centerness_loss': centerness_loss / B,
                'loss_conf': loss_conf / B}

        return predictions, batch_av_loss, losses_dict

    def forward(self, src, targets, masks):
        """
        :param x: (video(B, S, Dv), audio(B, S, Da), text(B, S, Dt))
        :param masks: {'V_mask':(B, 1, S), 'A_mask':(B, 1, S), 'T_mask':(B, 1, S), 'AV_mask':(B, 1, S)}
        :return: predictions, batch_av_loss, losses_dict
        """
        A = src['audio']
        V = src['rgb'] + src['flow']
        T = src['text']

        # １、特征预处理
        # (B, Sm, Dm)-->(B, Sm, Dm), m in (v,a,t)
        A = self.emb_A(A)
        V = self.emb_V(V)
        T = self.emb_T(T)
        A = self.pos_enc_A(A)
        V = self.pos_enc_V(V)
        T = self.pos_enc_T(T)

        # ２、特征融合编码
        Av, Va, Av_up, Va_up, AVT = self.encoder((A, V, T), masks)

        # 3、fcos实现proposal训练
        props_Av, loss_Av, losses_Av = self.fcos_prop(self.backbone_Av, self.fpn_Av, self.head_Av, Av_up, targets)
        props_Va, loss_Va, losses_Va = self.fcos_prop(self.backbone_Va, self.fpn_Va, self.head_Va, Va_up,targets)
        props_AVT, loss_AVT, losses_AVT = self.fcos_prop(self.backbone_AVT, self.fpn_AVT, self.head_AVT, AVT,targets)

        total_loss = 0.5*loss_Av + loss_Va + 0.5*loss_AVT

        # combine predictions,all_predictions=(B,10*48*800+10*128*300,2)
        all_predictions = torch.cat([props_Av, props_Va, props_AVT], dim=1)

        return all_predictions, total_loss, losses_Av, losses_Va, losses_AVT


def get_batch_position_annotations(cfg, preds, reg_heads, all_points_position, targets, timespan):
    """
    Assign a ground truth target for each position on feature map
    """
    #     device = annotations.device
    B, points_num, _ = preds.shape
    noobj_mask = torch.ones(B, points_num, device=preds.device).bool()
    obj_mask = torch.zeros_like(noobj_mask).bool()

    # EPS = 1e-16
    INF = 100000000
    mi = [[-1, 8], [8, 32], [32, 128], [128, 256], [256, 512], [512, 1024], [1024, 2048], [2048, INF]]
    batch_mi = []
    for reg_head, mi_i in zip(reg_heads, mi):
        mi_i = torch.tensor(mi_i).float()
        B, Si, _ = reg_head.shape
        per_level_mi = torch.zeros(B, Si, 2).float()
        per_level_mi = per_level_mi + mi_i
        batch_mi.append(per_level_mi)

    #     print('all_points_position:\n', all_points_position.shape)
    # [B, 2392, 2]
    all_points_mi = torch.cat(batch_mi, dim=1).to(cfg.device)

    # 对batch中每一个样本分别进行处理
    # 逐个样本对每一个位置分配目标标签，所有点的annotations，
    annotations = []
    for i in range(B):
        one_targets_idx = torch.nonzero(targets[:, 0] == i).squeeze()
        one_targets = targets[one_targets_idx, 1:3]
        annotations.append(one_targets)

    # ------------------print('annotations:\n', annotations)
    batch_targets = []
    # batch_positive_index = []
    for i, per_video_position, per_video_mi, per_video_annotations in zip(range(B), all_points_position, all_points_mi, annotations):
        per_video_gt_id = []
        # 每个图片的总点数
        points_num = per_video_position.shape[0]

        # 数据没有标注，也即数据没有对应的GT
        if per_video_annotations.shape[0] == 0:
            # 3:l,r，center-ness_gt
            per_video_targets = torch.zeros([points_num, 3])
        else:
            # 每个图片的GT数量
            annotation_num = per_video_annotations.shape[0]
            # 每个视频的proposal的start&end,并将其转换到grid网格中进行计算(gt_num,2)
            per_video_gt_bboxes = per_video_annotations / timespan

            # 建立总点数与总目标数的关系，生成候选值(points_num,gt_num,2)
            candidates = torch.zeros([points_num, annotation_num, 2], device=per_video_gt_bboxes.device)
            # 对于每一个位置点来说，标注的位置是不会变的
            candidates = candidates + per_video_gt_bboxes.unsqueeze(0)

            # (2392,1)-->(2392,gt_num,1)
            per_video_position = per_video_position.unsqueeze(1).repeat(
                1, annotation_num, 1).to(candidates.device)

            # 求每个位置点相对于GT框的l\r
            candidates[:, :, 0] = per_video_position[:, :, 0] - candidates[:, :, 0]
            candidates[:, :, 1] = candidates[:, :, 1] - per_video_position[:, :, 0]

            # 取出l\r中的最小值（2392,gt_num,2)--(2392,gt_num,1)
            candidates_min_value, _ = candidates.min(dim=-1, keepdim=True)

            # (2392,gt_num)-->(2392,gt_num,1)
            sample_flag = (candidates_min_value[:, :, 0] > 0).int().unsqueeze(-1)

            # get all negative reg targets which points ctr out of gt box,
            # 1、因为points不在目标框中，而被过滤掉一部分负样本
            candidates = candidates * sample_flag.float()

            # get all negative reg targets which assign ground turth not in range of mi
            # 取出l\r中的最大值
            candidates_max_value, _ = candidates.max(dim=-1, keepdim=True)
            candidates_max_value = candidates_max_value.to(per_video_mi.device)

            # （2392,2）-->(2392,gt_num,2)
            per_video_mi = per_video_mi.unsqueeze(1).repeat(1, annotation_num, 1)

            # 2、目标回归值大小限制，又过滤掉一部分负样本
            # (2392,5)-->(2392,5,1)
            m1_negative_flag = (candidates_max_value[:, :, 0] >=
                                per_video_mi[:, :, 0]).int().unsqueeze(-1)
            m1_negative_flag = m1_negative_flag.to(candidates.device)
            candidates = candidates * m1_negative_flag.float()

            m2_negative_flag = (candidates_max_value[:, :, 0] <
                                per_video_mi[:, :, 1]).int().unsqueeze(-1)
            m2_negative_flag = m2_negative_flag.to(candidates.device)
            candidates = candidates * m2_negative_flag.float()

            # (2392,5,2)-->(2392)
            final_sample_flag = candidates.sum(dim=-1).sum(dim=-1)
            final_sample_flag = final_sample_flag > 0

            # 获得正样本的索引,positive_index的尺寸不定，但是是一维的张量
            positive_index = (final_sample_flag == True).nonzero().squeeze(dim=-1)
            #             print('positive_index:\n', positive_index)

            # if no assign positive sample
            # 数据有标注，但是没有对应的正样本
            if len(positive_index) == 0:
                del candidates
                # 3:l,r,center-ness_gt
                per_video_targets = torch.zeros([points_num, 3], device=cfg.device).float()
            else:
                # 将正样本的目标值l/r取出来，第一维的尺度与positive_index相同（positive_num,gt_num,2)
                positive_candidates = candidates[positive_index]

                del candidates

                # 3:l,r,center-ness_gt
                # （2392,3）
                per_video_targets = torch.zeros([points_num, 3], device=cfg.device).float()

                # 处理可能出现的模糊样本
                for positive_candidate, positive_idx in zip(positive_candidates, positive_index):
                    # positive_candidate:(gt_num, 2)
                    # positive_idx:1
                    non_zero = torch.nonzero(positive_candidate)
                    #                     print('non_zero.shape:\n', non_zero)

                    # if only one candidate for each positive sample
                    # assign l,t,r,b,class_index,center_ness_gt ground truth
                    if non_zero.shape[0] == 2:
                        if non_zero[0, 0] in per_video_gt_id:
                            pass
                        else:
                            per_video_gt_id.append(non_zero[0, 0].item())
                        #                         print('positive_idx_one_by_one:\n', positive_idx, '\n')
                        #                         print('positive_candidate_one_by_one:\n', positive_candidate)
                        per_video_targets[positive_idx,
                        0:2] = positive_candidate[non_zero[0][0]]

                        l, r = per_video_targets[positive_index, 0], per_video_targets[positive_index, 1]

                        per_video_targets[positive_index, 2] = torch.sqrt((torch.min(l, r) / torch.max(l, r)))
                        # per_video_targets[positive_idx, 0:2] = torch.log(positive_candidate[non_zero[0][0]] + EPS)
                    else:
                        # if a positive point sample have serveral object candidates,
                        # then choose the smallest area object candidate as the ground turth for this positive point sample
                        #                         print('positive_candidate_one_by_many:\n', positive_candidate)
                        #                         print('positive_idx_one_by_many:\n', positive_idx, '\n')

                        #                         print('pisitive_candidate.shape:\n', positive_candidate.shape)
                        # (gt_num,2)
                        gts_center_l = (positive_candidate[:, 0] + positive_candidate[:, 1]) / 2

                        INF = 100000000
                        inf_tensor = torch.ones_like(gts_center_l) * INF
                        gts_center_l = torch.where(torch.eq(gts_center_l, 0.), inf_tensor, gts_center_l)

                        # print(gts_center_l.shape, per_video_gt_bboxes.shape)
                        gts_center = gts_center_l + per_video_gt_bboxes[:, 0]

                        points_local = per_video_position[positive_idx].squeeze(1)
                        distance = torch.sub(points_local, gts_center)
                        distance = torch.abs(distance)
                        _, min_index = distance.min(dim=0)

                        # 每个样本只匹配与GT中心点最近的那个GT（１，２）
                        final_candidate_reg_gts = positive_candidate[min_index, :]
                        if min_index in per_video_gt_id:
                            pass
                        else:
                            per_video_gt_id.append(min_index.item())
                        #                         print('final_candidate_reg_gts.shape:\n', final_candidate_reg_gts)
                        # assign l,t,r,b,class_index,center_ness_gt ground truth
                        per_video_targets[positive_idx, 0:2] = final_candidate_reg_gts

                        l, r = per_video_targets[positive_idx, 0], per_video_targets[positive_idx, 1]

                        per_video_targets[positive_idx, 2] = torch.sqrt((torch.min(l, r) / torch.max(l, r)))
                        # per_video_targets[positive_idx, 0:2] = torch.log(final_candidate_reg_gts + EPS)

            # positive_index = positive_index.cuda().data.cpu().numpy()
            # batch_positive_index.append(positive_index)
            obj_mask[i, positive_index] = 1
            noobj_mask[i,positive_index] = 0

        # (poimts_num, 3)-->（1，points_num, 3）
        # ------------------改进---------------------
        # per_video_targets[:, 0:2] = per_video_targets[:, 0:2] / strides
        # print(per_video_targets.shape)
        per_video_targets = per_video_targets.unsqueeze(0)
        batch_targets.append(per_video_targets)

    # (1,points_num, 3)-->(batch_size, points_num, 3)-->(batch_size, points_num, 4)
    batch_targets = torch.cat(batch_targets, dim=0).to(cfg.device)
    # batch_positive_id = np.array(batch_positive_index)
    all_points_position = all_points_position.to(batch_targets.device)
    batch_targets = torch.cat([batch_targets, all_points_position], dim=2)
    target_obj = obj_mask.float()

    # batch_targets shape:[batch_size, points_num, 4],4:l,r,center-ness_gt,point_ctr_x
    #     return reg_preds, center_preds, batch_targets, positive_index

    return batch_targets, obj_mask, noobj_mask, target_obj
