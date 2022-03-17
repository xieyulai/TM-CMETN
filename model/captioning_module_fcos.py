from copy import deepcopy

import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.blocks import (FeatureEmbedder, Identity,
                          PositionalEncoder, VocabularyEmbedder)
from model.decoders_fcos import TriModelDecoder
from model.encoders_fcos import TriModalEncoder, BiModalEncoderOne
from model.generators_fcos import GeneratorFCOS


class BiModalTransformer(nn.Module):

    def __init__(self, cfg, train_dataset):
        super(BiModalTransformer, self).__init__()
        self.d_model_A = cfg.d_model_audio
        self.d_model_V = cfg.d_model_video
        # self.d_model_T = cfg.d_model_text
        self.d_model = cfg.d_model
        self.dout_p = cfg.dout_p
        self.H = cfg.H
        self.N = cfg.N

        self.pad_idx = train_dataset.pad_idx

        if cfg.feature_fusion_mode == 'add':
            self.d_raw_caps = cfg.d_model_add
        elif cfg.feature_fusion_mode == 'cat':
            self.d_raw_caps = cfg.d_model_cat

        if cfg.use_linear_embedder:
            self.emb_A = FeatureEmbedder(cfg.d_aud, cfg.d_model_audio)
            self.emb_V = FeatureEmbedder(cfg.d_vid, cfg.d_model_video)
            # self.emb_T = FeatureEmbedder(cfg.d_text, cfg.d_model_text)
        else:
            self.emb_A = Identity()  # 128
            self.emb_V = Identity()  # 1024
            # self.emb_T = Identity()  # 300

        self.emb_C = VocabularyEmbedder(train_dataset.trg_voc_size, cfg.d_model_caps)  # (10172,300)

        # print('cfg.d_model_audio:\n',cfg.d_model_audio)          # 128
        # print('cfg.d_model_video:\n', cfg.d_model_video)         # 1024
        # print('cfg.d_model_caps:\n', cfg.d_model_caps)           # 300
        # 返回的是带有位置编码信息的特征矩阵
        self.pos_enc_A = PositionalEncoder(cfg.d_model_audio, cfg.dout_p)  # (32,*,128)
        self.pos_enc_V = PositionalEncoder(cfg.d_model_video, cfg.dout_p)  # (32,*,1024)
        # self.pos_enc_T = PositionalEncoder(cfg.d_model_text, cfg.dout_p)   # (32,*,300)
        self.pos_enc_C = PositionalEncoder(cfg.d_model_caps, cfg.dout_p)  # (32,*,300)

        # pdb.set_trace()
        self.encoder = BiModalEncoderOne(self.d_model_A, self.d_model_V, self.d_model, self.dout_p, self.H, self.N)

        self.decoder = TriModelDecoder(
            self.d_raw_caps, cfg.d_model_caps, cfg.d_model, cfg.dout_p,
            cfg.H, cfg.d_ff_caps, cfg.N
        )

        self.generator = GeneratorFCOS(cfg.d_model_caps, train_dataset.trg_voc_size)

        print('initialization: xavier')
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # initialize embedding after, so it will replace the weights
        # of the prev. initialization
        # 将word index转化为词向量
        self.emb_C.init_word_embeddings(train_dataset.train_vocab.vectors, cfg.unfreeze_word_emb)

        # load the pre_trained encoder from the proposal (used in ablation studies)
        if cfg.pretrained_prop_model_path is not None:
            print(f'Pre_trained prop path: \n {cfg.pretrained_prop_model_path}')
            cap_model_cpt = torch.load(cfg.pretrained_prop_model_path, map_location='cpu')
            encoder_config = cap_model_cpt['config']
            self.encoder = BiModalEncoderOne(encoder_config.d_model_audio, encoder_config.d_model_video,
                                             encoder_config.d_model, encoder_config.dout_p, encoder_config.H, encoder_config.N)
            encoder_weights = {k: v for k, v in cap_model_cpt['model_state_dict'].items() if 'encoder' in k}
            encoder_weights = {k.replace('encoder.', ''): v for k, v in encoder_weights.items()}
            self.encoder.load_state_dict(encoder_weights)
            self.encoder = self.encoder.to(cfg.device)
            for param in self.encoder.parameters():
                param.requires_grad = cfg.finetune_prop_encoder

    def forward(self, src: dict, trg, masks: dict):
        # V.shape=(32,*,1024),A.shape=(32,*,128)
        A, V, T = src  # 将rgb和flow两个特征线性相加作为V的特征
        C = trg
        # print('V.shape,A.shape,T.shape,C.shape:\n', V.shape, A.shape, C.shape)
        # print('V,A,C:\n', V[0], A[0], C[0])

        # (B, Sm, Dm) <- (B, Sm, Dm), m in [a, v];
        A = self.emb_A(A)
        V = self.emb_V(V)
        # T = self.emb_T(T)
        # (B, Sc, Dc) <- (S, Sc)
        C = self.emb_C(C)

        A = self.pos_enc_A(A)
        V = self.pos_enc_V(V)
        # T = self.pos_enc_T(T)
        C = self.pos_enc_C(C)
        # print('WordEmbedding:\n', C[0][0])   # batch中，第一个caption的第一个词嵌入向量表示

        # notation: M1m2m2 (B, Sm1, Dm1), M1 is the target modality, m2 is the source modality
        # Av--M1m2 (B, Sm1, Dm1), Va--M2m1 (B, Sm2, Dm2)
        Av, Va = self.encoder((A, V), masks)

        AV = torch.add(Av, Va)

        # (B, Sc, Dc)
        C = self.decoder((C, AV), masks)

        # (B, Sc, Vocabc) <- (B, Sc, Dc)
        C = self.generator(C)

        return C


class TriModalTransformer(nn.Module):

    def __init__(self, cfg, train_dataset):
        super(TriModalTransformer, self).__init__()
        self.cfg = cfg
        self.pad_idx = train_dataset.pad_idx

        if cfg.AVT_fusion_mode == 'add':
            self.d_raw_caps = cfg.d_model//4
        else:
            self.d_raw_caps = cfg.d_model//2

        self.emb_A = Identity()
        self.emb_V = Identity()
        self.emb_T = Identity()

        self.emb_C = VocabularyEmbedder(train_dataset.trg_voc_size, cfg.d_model_caps)  # (10172,300)
        self.pos_enc_A = PositionalEncoder(cfg.d_model_audio, cfg.dout_p)  # (32,*,128)
        self.pos_enc_V = PositionalEncoder(cfg.d_model_video, cfg.dout_p)  # (32,*,1024)
        self.pos_enc_T = PositionalEncoder(cfg.d_model_text, cfg.dout_p)   # (32,*,300)
        self.pos_enc_C = PositionalEncoder(cfg.d_model_caps, cfg.dout_p)  # (32,*,300)

        self.encoder = TriModalEncoder(cfg, cfg.d_model_audio, cfg.d_model_video, cfg.d_model_text, cfg.d_model, cfg.dout_p, cfg.d_ff_audio, cfg.d_ff_video, cfg.d_ff_text)

        self.decoder = TriModelDecoder(self.d_raw_caps, cfg.d_model_caps, cfg.d_model, cfg.dout_p, cfg.H, cfg.N, cfg.d_ff_cap)

        self.generator = GeneratorFCOS(cfg.d_model_caps, train_dataset.trg_voc_size)

        print('initialization: xavier')
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.emb_C.init_word_embeddings(train_dataset.train_vocab.vectors, cfg.unfreeze_word_emb)

        if cfg.pretrained_prop_model_path is not None:
            print(f'Pre_trained prop path: \n {cfg.pretrained_prop_model_path}')
            cap_model_cpt = torch.load(cfg.pretrained_prop_model_path, map_location='cpu')
            pre_cfg = cap_model_cpt['config']
            self.encoder = TriModalEncoder(pre_cfg, pre_cfg.d_model_audio, pre_cfg.d_model_video,
                                           pre_cfg.d_model_text, pre_cfg.d_model, pre_cfg.dout_p)
            encoder_weights = {k: v for k, v in cap_model_cpt['model_state_dict'].items() if 'encoder' in k}
            encoder_weights = {k.replace('encoder.', ''): v for k, v in encoder_weights.items()}
            self.encoder.load_state_dict(encoder_weights)
            self.encoder = self.encoder.to(cfg.device)
            for param in self.encoder.parameters():
                param.requires_grad = cfg.finetune_prop_encoder

    def forward(self, src: dict, trg, masks: dict):
        A = src['audio']
        V = src['rgb'] + src['flow']
        T = src['text']
        C = trg

        A = self.emb_A(A)
        V = self.emb_V(V)
        T = self.emb_T(T)
        C = self.emb_C(C)

        A = self.pos_enc_A(A)
        V = self.pos_enc_V(V)
        T = self.pos_enc_T(T)
        C = self.pos_enc_C(C)

        Av, Va, AVT = self.encoder((A, V, T), masks)

        C = self.decoder((C, AVT), masks)

        C = self.generator(C)

        return C
