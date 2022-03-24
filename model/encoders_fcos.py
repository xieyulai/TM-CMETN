import torch
import torch.nn as nn
import torch.nn.functional as F

from model.blocks import LayerStack, PositionwiseFeedForward, ResidualConnection, clone, PositionalEncoder
from model.multihead_attention import MultiHeadedAttention
from model.masking import upsample


class TriModalEncoder(nn.Module):

    def __init__(self, cfg, d_model_A, d_model_V, d_model_T, d_model, dout_p, d_ff_A, d_ff_V, d_ff_T):
        super(TriModalEncoder, self).__init__()
        self.cfg = cfg
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
        self.learn_param = nn.Parameter(data=torch.FloatTensor([1.0]).to(cfg.device), requires_grad=True)

    def forward(self, x, masks):
        """
        :param x: x: (audio(B, S, Da), video(B, S, Dv), text(B, S, Dt))
        :param masks: {'V_mask':(B, 1, S), 'A_mask':(B, 1, S), 'T_mask':(B, 1, S)}
        :return: avt:(B, S, D_fcos)
        """
        A, V, T = x

        Av, Va = self.encoder_one((A, V), masks)

        # if self.procedure == 'train_prop':
        Av = upsample(Av, self.cfg.scale_audio)
        Va = upsample(Va, self.cfg.scale_video)

        if self.procedure == 'train_cap':
            if Av.shape[1] == Va.shape[1]:
                pass
            elif Av.shape[1] < Va.shape[1]:
                s = Va.shape[1] - Av.shape[1]
                p1d = [0, 0, 0, s]
                Av = F.pad(Av, p1d, value=0)
            elif Av.shape[1] > Va.shape[1]:
                s = Av.shape[1] - Va.shape[1]
                p1d = [0, 0, 0, s]
                Va = F.pad(Va, p1d, value=0)
        ## A和V的融合
        if self.cfg.AV_fusion_mode == 'add':
            AV = torch.add(Av, Va)
        else:
            AV = torch.cat((Av, Va), dim=-1)

        ## proposal降维
        AV = AV.permute(0, 2, 1)
        if self.procedure == 'train_prop':
            AV = self.conv_enc_av_1(AV)
            AV = self.conv_enc_av_2(AV)
        AV = AV.permute(0, 2, 1)
        AV = self.pos_enc_mid(AV)

        AVt, Tav = self.encoder_tow((AV, T), masks)

        ## T分支上加可学习参数
        Tav = Tav * self.learn_param

        if self.procedure == 'train_cap':
            if AVt.shape[1] == Tav.shape[1]:
                pass
            elif AVt.shape[1] < Tav.shape[1]:
                s = Tav.shape[1] - AVt.shape[1]
                p1d = [0, 0, 0, s]
                AVt = F.pad(AVt, p1d, value=0)
            elif AVt.shape[1] > Tav.shape[1]:
                s = AVt.shape[1] - Tav.shape[1]
                p1d = [0, 0, 0, s]
                Tav = F.pad(Tav, p1d, value=0)
        else:
            AVt = upsample(AVt, 4)
        ## AV和T的融合
        if self.cfg.AVT_fusion_mode == 'add':
            AVT = torch.add(AVt, Tav)
        else:
            AVT = torch.cat((AVt, Tav), dim=-1)

        ## caption 降维
        AVT = AVT.permute(0, 2, 1)
        if self.cfg.procedure == 'train_cap':
            AVT = self.conv_enc_two(AVT)
        AVT = AVT.permute(0, 2, 1)

        return Av, Va, AVT


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

    def __init__(self, d_model_A, d_model_V, d_model, dout_p, H, N, d_ff_A, d_ff_V, uni_dim='conv'):
        super(BiModalEncoderOne, self).__init__()
        self.uni_dim = uni_dim
        layer_AV = BiModalEncoderLayer(d_model_A, d_model_V, d_model, dout_p, H, d_ff_A, d_ff_V)
        self.encoder_AV = LayerStack(layer_AV, N)  # N=2

        ## 线性层统一维度
        if self.uni_dim == 'linear':
            self.linear_A = nn.Linear(d_model_A, d_model//4)
            self.linear_V = nn.Linear(d_model_V, d_model//4)
        ## 1维卷积, 参数量少
        else:
            self.convd_A = nn.Conv1d(d_model_A, d_model//4, kernel_size=1, stride=1, padding=0)
            self.convd_V = nn.Conv1d(d_model_V, d_model//4, kernel_size=1, stride=1, padding=0)

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

        if self.uni_dim == 'linear':
            Av, Va = self.linear_A(Av), self.linear_V(Va)
        else:
            Av, Va = self.convd_A(Av.permute(0, 2, 1)), self.convd_V(Va.permute(0, 2, 1))
            Av, Va = Av.permute(0, 2, 1), Va.permute(0, 2, 1)

        return Av, Va


class BiModalEncoderTow(nn.Module):

    def __init__(self, d_model_AV, d_model_T, d_model, dout_p, H, N, d_ff_AV, d_ff_T, uni_dim='conv'):
        super(BiModalEncoderTow, self).__init__()
        self.uni_dim = uni_dim
        layer_AVT = BiModalEncoderLayer(d_model_AV, d_model_T, d_model, dout_p, H, d_ff_AV, d_ff_T)
        self.encoder_AVT = LayerStack(layer_AVT, N)  # N=2

        ## 统一维度
        if self.uni_dim == 'linear':
            self.linear_AV = nn.Linear(d_model_AV, d_model//4)
            self.linear_T = nn.Linear(d_model_T, d_model//4)
        ## 1维卷积, 参数量少
        else:
            self.convd_AV = nn.Conv1d(d_model_AV, d_model//4, kernel_size=1, stride=1, padding=0)
            self.convd_T = nn.Conv1d(d_model_T, d_model//4, kernel_size=1, stride=1, padding=0)

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

        if self.uni_dim == 'linear':
            AVt, Tav = self.linear_AV(AVt), self.linear_T(Tav)
        else:
            AVt, Tav = self.convd_AV(AVt.permute(0, 2, 1)), self.convd_T(Tav.permute(0, 2, 1))
            AVt, Tav = AVt.permute(0, 2, 1), Tav.permute(0, 2, 1)

        return AVt, Tav
