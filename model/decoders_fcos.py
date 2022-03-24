import torch
import torch.nn as nn

from model.blocks import (LayerStack, PositionwiseFeedForward, ResidualConnection, BridgeConnection)
from model.multihead_attention import MultiHeadedAttention


class TriModalDecoderLayer(nn.Module):

    def __init__(self, d_model_AVT, d_model_V, d_model_C, d_model, dout_p, H, d_ff_C):
        super(TriModalDecoderLayer, self).__init__()

        self.res_layer_self_att = ResidualConnection(d_model_C, dout_p)
        self.self_att = MultiHeadedAttention(d_model_C, d_model_C, d_model_C, H, d_model, dout_p)

        self.res_layer_enc_att_AVT = ResidualConnection(d_model_C, dout_p)
        self.enc_att_AVT = MultiHeadedAttention(d_model_C, d_model_AVT, d_model_AVT, H, d_model, dout_p)
        self.res_layer_enc_att_V = ResidualConnection(d_model_C, dout_p)
        self.enc_att_V = MultiHeadedAttention(d_model_C, d_model_V, d_model_V, H, d_model, dout_p)

        self.bridge = BridgeConnection(2*d_model_C, d_model_C, dout_p)

        self.res_layer_ff = ResidualConnection(d_model_C, dout_p)
        self.feed_forward = PositionwiseFeedForward(d_model_C, d_ff_C, dout_p)

    def forward(self, x, masks):
        '''
        Inputs:
            x (C, memory): C: (B, Sc, Dc)
                           memory: AVT: (B, Savt, d_model)
            masks (AVT_mask: (B, 1, Savt); C_mask (B, Sc, Sc))
        Outputs:
            x (C, memory): C: (B, Sc, Dc)
                           memory: AVT: (B, Savt, Davt)
        '''
        C, memory = x
        V, AVT = memory

        def sublayer_self_att(C): return self.self_att(C, C, C, masks['C_mask'])
        def sublayer_enc_att_AVT(C): return self.enc_att_AVT(C, AVT, AVT, None)
        def sublayer_enc_att_V(C): return self.enc_att_AVT(C, V, V, None)
        sublayer_feed_forward = self.feed_forward

        C = self.res_layer_self_att(C, sublayer_self_att)

        Cavt = self.res_layer_enc_att_AVT(C, sublayer_enc_att_AVT)
        Cv = self.res_layer_enc_att_AVT(C, sublayer_enc_att_V)

        C = torch.cat([Cavt, Cv], dim=-1)
        # bridge: (B, Sc, Dc) <- (B, Sc, 2*Dc)
        C = self.bridge(C)

        C = self.res_layer_ff(C, sublayer_feed_forward)

        return C, memory


class TriModelDecoder(nn.Module):

    def __init__(self, d_raw_caps, d_model_V, d_model_C, d_model, dout_p, H, N, d_ff_C):
        super(TriModelDecoder, self).__init__()
        layer = TriModalDecoderLayer(d_raw_caps, d_model_V, d_model_C, d_model, dout_p, H, d_ff_C)
        self.decoder = LayerStack(layer, N)

    def forward(self, x, masks):
        '''
        Inputs:
            x (C, memory): C: (B, Sc, Dc)
                           memory: (Av: (B, Sa, Da), Va: (B, Sv, Dv))
            masks (V_mask: (B, 1, Sv); A_mask: (B, 1, Sa); C_mask (B, Sc, Sc))
        Outputs:
            x (C, memory): C: (B, Sc, Dc)
                memory: (Av: (B, Sa, Da), Va: (B, Sv, Dv))
        '''
        C, memory = self.decoder(x, masks)

        return C
