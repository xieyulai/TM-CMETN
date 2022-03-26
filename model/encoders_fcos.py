import torch
import torch.nn as nn
import torch.nn.functional as F

from model.blocks import LayerStack, PositionwiseFeedForward, ResidualConnection, clone, PositionalEncoder
from model.multihead_attention import MultiHeadedAttention
from model.masking import upsample


class TriModalEncoder(nn.Module):

    def __init__(self, cfg, d_model_A, d_model_V, d_model_T, d_model, dout_p, d_ff_A, d_ff_V, d_ff_T, uni_dim='conv'):
        super(TriModalEncoder, self).__init__()
        self.cfg = cfg
        self.uni_dim = uni_dim
        self.procedure = cfg.procedure
        self.modality = cfg.modality
        self.d_model_A = d_model_A
        self.d_model_V = d_model_V
        self.d_model_T = d_model_T
        self.d_model = d_model
        self.dout_p = dout_p
        self.d_ff_A = d_ff_A
        self.d_ff_V = d_ff_V
        self.d_ff_T = d_ff_T
        self.H = cfg.H
        self.N = cfg.N

        if cfg.AV_fusion_mode == 'add':
            self.d_model_mid = cfg.d_model//4
        else:
            self.d_model_mid = cfg.d_model//2

        if cfg.AVT_fusion_mode == 'add':
            self.d_raw_caps = cfg.d_model//4
        else:
            self.d_raw_caps = cfg.d_model//2

        ## 线性层统一维度
        if self.uni_dim == 'linear':
            self.linear_A = nn.Linear(self.d_model_A, self.d_model//4)
            self.linear_V = nn.Linear(self.d_model_V, self.d_model//4)
            self.linear_AV = nn.Linear(self.d_model_mid, self.d_model//4)
            self.linear_T = nn.Linear(self.d_model_T, self.d_model//4)
        ## 1维卷积, 参数量少
        else:
            self.convd_A = nn.Conv1d(self.d_model_A, self.d_model//4, kernel_size=1, stride=1, padding=0)
            self.convd_V = nn.Conv1d(self.d_model_V, self.d_model//4, kernel_size=1, stride=1, padding=0)
            self.convd_AV = nn.Conv1d(self.d_model_mid, self.d_model//4, kernel_size=1, stride=1, padding=0)
            self.convd_T = nn.Conv1d(self.d_model_T, self.d_model//4, kernel_size=1, stride=1, padding=0)

        self.pos_enc_mid = PositionalEncoder(self.d_model_mid, cfg.dout_p)

        self.conv_enc_av_1 = nn.Conv1d(self.d_model_mid, self.d_model_mid, kernel_size=3, stride=2, padding=1)
        self.conv_enc_av_2 = nn.Conv1d(self.d_model_mid, self.d_model_mid, kernel_size=3, stride=2, padding=1)
        self.conv_enc_two = nn.Conv1d(self.d_raw_caps, self.d_raw_caps, kernel_size=3, stride=2, padding=1)

        self.encoder_one = BiModalEncoderOne(self.d_model_A, self.d_model_V, self.d_model,
                                             self.dout_p, self.H, self.N, self.d_ff_A, self.d_ff_V)
        self.encoder_tow = BiModalEncoderTow(self.d_model_mid, self.d_model_T, self.d_model,
                                             self.dout_p, self.H, self.N, self.d_model_mid*4, self.d_ff_T)
        # if cfg.dataset_type == 2000:
        #     self.learn_param = nn.Parameter(data=torch.FloatTensor([1.0]).to(cfg.device), requires_grad=False)
        # else:
        self.learn_param = nn.Parameter(data=torch.FloatTensor([1.0]).to(cfg.device), requires_grad=False)

    def forward(self, x, masks):
        """
        :param x: x: (audio(B, S, Da), video(B, S, Dv), text(B, S, Dt))
        :param masks: {'V_mask':(B, 1, S), 'A_mask':(B, 1, S), 'T_mask':(B, 1, S)}
        :return: avt:(B, S, D_fcos)
        """
        A, V, T = x

        ## 1st CM-ENC
        Av, Va = self.encoder_one((A, V), masks)

        ## 统一特征维度
        if self.uni_dim == 'linear':
            Av_uni, Va_uni = self.linear_A(Av), self.linear_V(Va)
        else:
            Av_uni, Va_uni = self.convd_A(Av.permute(0, 2, 1)), self.convd_V(Va.permute(0, 2, 1))
            Av_uni, Va_uni = Av_uni.permute(0, 2, 1), Va_uni.permute(0, 2, 1)

        ## 时序语义对齐
        Av_up = upsample(Av_uni, self.cfg.scale_audio)
        Va_up = upsample(Va_uni, self.cfg.scale_video)

        if self.procedure == 'train_cap':
            if Av_up.shape[1] == Va_up.shape[1]:
                pass
            elif Av_up.shape[1] < Va_up.shape[1]:
                s = Va_up.shape[1] - Av_up.shape[1]
                p1d = [0, 0, 0, s]
                Av_up = F.pad(Av_up, p1d, value=0)
            elif Av_up.shape[1] > Va_up.shape[1]:
                s = Av_up.shape[1] - Va_up.shape[1]
                p1d = [0, 0, 0, s]
                Va_up = F.pad(Va_up, p1d, value=0)

        ## A和V的融合
        if self.cfg.AV_fusion_mode == 'add':
            AV_fus = torch.add(Av_up, Va_up)
        else:
            AV_fus = torch.cat((Av_up, Va_up), dim=-1)

        ## proposal降维
        AV_fus = AV_fus.permute(0, 2, 1)
        if self.procedure == 'train_prop':
            AV_fus = self.conv_enc_av_1(AV_fus)
            AV_fus = self.conv_enc_av_2(AV_fus)
        AV_fus = AV_fus.permute(0, 2, 1)
        AV_fus = self.pos_enc_mid(AV_fus)

        ## 2nd CM-ENC
        AVt, Tav = self.encoder_tow((AV_fus, T), masks)

        ## 统一特征维度
        if self.uni_dim == 'linear':
            AVt_uni, Tav_uni = self.linear_AV(AVt), self.linear_T(Tav)
        else:
            AVt_uni, Tav_uni = self.convd_AV(AVt.permute(0, 2, 1)), self.convd_T(Tav.permute(0, 2, 1))
            AVt_uni, Tav_uni = AVt_uni.permute(0, 2, 1), Tav_uni.permute(0, 2, 1)

        ## T分支上加可学习参数
        Tav_uni = Tav_uni * self.learn_param

        if self.procedure == 'train_cap':
            if AVt_uni.shape[1] == Tav_uni.shape[1]:
                pass
            elif AVt_uni.shape[1] < Tav_uni.shape[1]:
                s = Tav_uni.shape[1] - AVt_uni.shape[1]
                p1d = [0, 0, 0, s]
                AVt_uni = F.pad(AVt_uni, p1d, value=0)
            elif AVt_uni.shape[1] > Tav_uni.shape[1]:
                s = AVt_uni.shape[1] - Tav_uni.shape[1]
                p1d = [0, 0, 0, s]
                Tav_uni = F.pad(Tav_uni, p1d, value=0)
        else:
            AVt_uni = upsample(AVt_uni, 4)

        ## AV和T的融合
        if self.cfg.AVT_fusion_mode == 'add':
            AVT_fus = torch.add(AVt_uni, Tav_uni)
        else:
            AVT_fus = torch.cat((AVt_uni, Tav_uni), dim=-1)

        ## caption降维
        AVT_fus = AVT_fus.permute(0, 2, 1)
        if self.cfg.procedure == 'train_cap':
            AVT_fus = self.conv_enc_two(AVT_fus)
        AVT_fus = AVT_fus.permute(0, 2, 1)

        return Av, Va, Av_up, Va_up, AVT_fus


class BiModalEncoderLayer(nn.Module):

    def __init__(self, d_model_M1, d_model_M2, d_model, dout_p, H, d_ff_A, d_ff_V):
        super(BiModalEncoderLayer, self).__init__()
        self.self_att_M1 = MultiHeadedAttention(d_model_M1, d_model_M1, d_model_M1, H, d_model, dout_p)
        self.self_att_M2 = MultiHeadedAttention(d_model_M2, d_model_M2, d_model_M2, H, d_model, dout_p)

        self.cross_att_M1 = MultiHeadedAttention(d_model_M1, d_model_M2, d_model_M2, H, d_model, dout_p)
        self.cross_att_M2 = MultiHeadedAttention(d_model_M2, d_model_M1, d_model_M1, H, d_model, dout_p)

        self.feed_forward_M1 = PositionwiseFeedForward(d_model_M1, d_ff_A, dout_p)
        self.feed_forward_M2 = PositionwiseFeedForward(d_model_M2, d_ff_V, dout_p)

        self.res_layers_M1 = clone(ResidualConnection(d_model_M1, dout_p), 3)
        self.res_layers_M2 = clone(ResidualConnection(d_model_M2, dout_p), 3)

    def forward(self, x, masks):
        '''
        Forward:
            x:(A,V)
                A='audio' (B, Sa, Da)、  V='rgb'&'flow' (B, Sv, Dv),
            masks:(A_mask,V_mask)
                A_mask (B, 1, Sa), V_mask (B, 1, Sv)
            Output:
                Av:(B, Sa, Da), Va:(B, Sv, Da)
        '''
        M1, M2 = x
        M1_mask, M2_mask = masks

        def sublayer_self_att_M1(M1): return self.self_att_M1(M1, M1, M1, M1_mask)
        def sublayer_self_att_M2(M2): return self.self_att_M2(M2, M2, M2, M2_mask)
        def sublayer_att_M1(M1): return self.cross_att_M1(M1, M2, M2, M2_mask)
        def sublayer_att_M2(M2): return self.cross_att_M2(M2, M1, M1, M1_mask)

        sublayer_ff_M1 = self.feed_forward_M1
        sublayer_ff_M2 = self.feed_forward_M2

        M1 = self.res_layers_M1[0](M1, sublayer_self_att_M1)
        M2 = self.res_layers_M2[0](M2, sublayer_self_att_M2)

        M1m2 = self.res_layers_M1[1](M1, sublayer_att_M1)
        M2m1 = self.res_layers_M2[1](M2, sublayer_att_M2)

        M1m2 = self.res_layers_M1[2](M1m2, sublayer_ff_M1)
        M2m1 = self.res_layers_M2[2](M2m1, sublayer_ff_M2)

        return M1m2, M2m1


class BiModalEncoderOne(nn.Module):

    def __init__(self, d_model_A, d_model_V, d_model, dout_p, H, N, d_ff_A, d_ff_V):
        super(BiModalEncoderOne, self).__init__()
        layer_AV = BiModalEncoderLayer(d_model_A, d_model_V, d_model, dout_p, H, d_ff_A, d_ff_V)
        self.encoder_AV = LayerStack(layer_AV, N)  # N=2

    def forward(self, x, masks: dict):
        '''
        Input:
            x (A, V): (B, Sm, D)
            masks: {V_mask: (B, 1, Sv); A_mask: (B, 1, Sa)}
        Output:
            (Av, Va): (B, Sm1, Dm1)
        '''
        A, V = x

        Av, Va = self.encoder_AV((A, V), (masks['A_mask'], masks['V_mask']))

        return Av, Va


class BiModalEncoderTow(nn.Module):

    def __init__(self, d_model_AV, d_model_T, d_model, dout_p, H, N, d_ff_AV, d_ff_T):
        super(BiModalEncoderTow, self).__init__()
        layer_AVT = BiModalEncoderLayer(d_model_AV, d_model_T, d_model, dout_p, H, d_ff_AV, d_ff_T)
        self.encoder_AVT = LayerStack(layer_AVT, N)  # N=2

    def forward(self, x, masks: dict):
        '''
        Input:
            x (A, V): (B, Sm, D)
            masks: {V_mask: (B, 1, Sv); A_mask: (B, 1, Sa)}
        Output:
            (Av, Va): (B, Sm1, Dm1)
        '''
        AV, T = x

        AVt, Tav = self.encoder_AVT((AV, T), (None, masks['T_mask']))

        return AVt, Tav
