import os
from time import localtime, strftime


class Config(object):
    '''
    Note: don't change the methods of this class later in code.
    '''

    def __init__(self, args):
        '''
        Try not to create anything here: like new forders or something
        '''
        self.curr_time = strftime('%y%m%d%H%M%S', localtime())

        self.procedure = args.procedure
        self.dataset_type = args.dataset_type
        # dataset
        if args.dataset_type == 2000:
            self.train_meta_path = args.train_path_2000
            self.val_1_meta_path = args.val_1_path_2000
            self.val_2_meta_path = args.val_2_path_2000
        else:
            self.train_meta_path = args.train_path
            self.val_1_meta_path = args.val_1_path
            self.val_2_meta_path = args.val_2_path

        self.modality = args.modality
        self.video_feature_name = args.video_feature_name
        self.audio_feature_name = args.audio_feature_name
        self.text_feature_name = args.text_feature_name
        self.video_features_path = args.video_features_path
        self.audio_features_path = args.audio_features_path
        self.align_text_features_path = args.align_text_features_path

        self.d_vid = args.d_vid
        self.d_aud = args.d_aud
        self.d_text = args.d_text
        self.scale_audio = args.scale_audio
        self.scale_video = args.scale_video
        self.start_token = args.start_token
        self.end_token = args.end_token
        self.pad_token = args.pad_token
        self.max_len = args.max_len
        self.min_freq_caps = args.min_freq_caps

        # model
        self.is_scale = args.is_scale
        self.AV_fusion_mode = args.AV_fusion_mode
        self.AVT_fusion_mode = args.AVT_fusion_mode
        if args.procedure == 'train_cap' or args.procedure == 'eval_caption':
            self.word_emb_caps = args.word_emb_caps
            self.unfreeze_word_emb = args.unfreeze_word_emb
            self.model = args.model
            self.pretrained_prop_model_path = args.pretrained_prop_model_path
            self.finetune_prop_encoder = args.finetune_prop_encoder
            self.inherit_cap_model_path = args.inherit_cap_model_path
        elif args.procedure == 'train_prop':
            self.word_emb_caps = args.word_emb_caps  # ActivityNetCaptionsDataset() needs it
            self.pretrained_cap_model_path = args.pretrained_cap_model_path
            self.finetune_cap_encoder = args.finetune_cap_encoder
            self.layer_norm = args.layer_norm
            self.noobj_coeff = args.noobj_coeff
            self.obj_coeff = args.obj_coeff
            self.cen_coeff = args.cen_coeff
            self.reg_coeff = args.reg_coeff
            self.nms_tiou_thresh = args.nms_tiou_thresh
            self.strides = {}
            self.pad_feats_up_to = {}
            if 'audio' in self.modality:
                self.strides['audio'] = args.audio_feature_timespan
                self.pad_feats_up_to['audio'] = args.pad_audio_feats_up_to
            if 'video' in self.modality:
                self.feature_timespan_in_fps = args.feature_timespan_in_fps
                self.fps_at_extraction = args.fps_at_extraction
                self.strides['video'] = args.feature_timespan_in_fps / args.fps_at_extraction
                self.pad_feats_up_to['video'] = args.pad_video_feats_up_to
            if 'text' in self.modality:
                self.pad_feats_up_to['text'] = args.pad_text_feats_up_to
            self.timespan_fcos = args.timespan_fcos
            self.strides_fcos = args.strides_fcos
            self.fpn_feature_sizes = args.fpn_feature_sizes
            self.planes = args.planes
            self.C3_inplanes = args.C3_inplanes
            self.C4_inplanes = args.C4_inplanes
            self.C5_inplanes = args.C5_inplanes
        elif args.procedure == 'evaluate':
            self.pretrained_cap_model_path = args.pretrained_cap_model_path
        else:
            raise NotImplementedError

        self.pretrained_bmt_model_path = args.pretrained_bmt_model_path
        self.finetune_bmt_encoder = args.finetune_bmt_encoder

        self.dout_p = args.dout_p
        self.dout_p_fcos = args.dout_p_fcos
        self.N = args.N
        self.H = args.H
        self.d_model = args.d_model

        # if args.use_linear_embedder:
        #     self.d_model_video = self.d_model
        #     self.d_model_audio = self.d_model
        #     self.d_model_text = self.d_model
        # else:
        self.d_model_video = self.d_vid
        self.d_model_audio = self.d_aud
        self.d_model_text = self.d_text

        self.d_model_caps = args.d_model_caps
        if 'video' in self.modality:
            self.d_ff_video = 4*self.d_model_video if args.d_ff_video is None else args.d_ff_video
        if 'audio' in self.modality:
            self.d_ff_audio = 4*self.d_model_audio if args.d_ff_audio is None else args.d_ff_audio
        if 'text' in self.modality:
            self.d_ff_text = 4*self.d_model_text if args.d_ff_text is None else args.d_ff_text
        self.d_ff_cap = 4*self.d_model_caps if args.d_ff_cap is None else args.d_ff_cap
        # training
        self.device_ids = (args.device_ids)                # [0,1,2]
        self.device = f'cuda:{self.device_ids[0]}'     # cuda:0
        self.train_batch_size = args.B * len(self.device_ids)     # 32
        self.inference_batch_size = args.inf_B_coeff * self.train_batch_size   # 64
        self.epoch_num = args.epoch_num
        self.one_by_one_starts_at = args.one_by_one_starts_at
        self.early_stop_after = args.early_stop_after
        # criterion
        self.smoothing = args.smoothing  # 0 == cross entropy
        self.grad_clip = args.grad_clip
        # lr
        self.lr = args.lr
        self.milestones = args.milestones
        self.gamma = args.gamma
        # evaluation
        if args.dataset_type == 2000:
            self.val_reference_paths = args.reference_paths_2000
        elif args.dataset_type == 9000:
            self.val_reference_paths = args.reference_paths
        else:
            raise Exception('Datasets setting error!')
        self.tIoUs = args.tIoUs
        self.max_prop_per_vid = args.max_prop_per_vid
        self.prop_pred_path = args.prop_pred_path     # 需要传入的参数,val_1_max100*.json
        self.val_prop_meta_path = args.val_prop_meta_path
        self.avail_mp4_path = args.avail_mp4_path
        # logging
        self.to_log = args.to_log
        if args.to_log:
            self.log_dir = os.path.join(args.log_dir, args.procedure)
            self.checkpoint_dir = os.path.join(args.checkpoint_dir, args.procedure)
            exper_name = f'{self.curr_time[2:]}_{self.modality}'
            self.log_path = os.path.join(self.log_dir, exper_name)
            self.model_checkpoint_path = os.path.join(self.checkpoint_dir, exper_name)
        else:
            self.log_dir = None
            self.log_path = None
        self.debug = args.debug
        self.keep_train = args.keep_train
