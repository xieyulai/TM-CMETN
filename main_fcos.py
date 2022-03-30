import argparse
from pprint import pprint
import os
import shutil

from utilities.config_constructor_fcos import Config
from scripts.train_captioning_module_fcos import train_cap_fcos
from scripts.train_proposal_generator_fcos import train_prop_fcos
from scripts.eval_on_learned_props_fcos import eval_on_learned_props_fcos


def main(cfg):
    if cfg.procedure != 'evaluate':
        os.makedirs(cfg.log_path)
        oldname = os.getcwd() + os.sep
        newname = cfg.log_path + os.sep
        shutil.copyfile(oldname + 'main_fcos.py', newname + 'main_fcos.py')
        shutil.copytree(oldname + 'datasets', newname + 'datasets')
        shutil.copytree(oldname + 'epoch_loops', newname + 'epoch_loops')
        shutil.copytree(oldname + 'evaluation', newname + 'evaluation')
        shutil.copytree(oldname + 'model', newname + 'model')
        shutil.copytree(oldname + 'scripts', newname + 'scripts')
        shutil.copytree(oldname + 'utilities', newname + 'utilities')

    if cfg.procedure == 'train_cap':
        shutil.copyfile(oldname + 'run_caption.sh', newname + 'run_caption.sh')
        train_cap_fcos(cfg)
    elif cfg.procedure == 'train_prop':
        shutil.copyfile(oldname + 'run_proposal.sh', newname + 'run_proposal.sh')
        train_prop_fcos(cfg)
    elif cfg.procedure == 'evaluate':
        eval_on_learned_props_fcos(cfg)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    '''
        Note, that the arguments are shared for both train_cap and train_prop that leads to the 
        situation in which an argument is defined but unused (--word_emb_caps for train_prop case).
    '''
    parser = argparse.ArgumentParser(description='Run experiment')

    ## DATA
    # paths to the precalculated train meta files
    parser.add_argument('--dataset_type', type=int, default=2000, help='train model datasets size selection')
    parser.add_argument('--train_path', type=str, default='./data/train_no_missings.csv')
    parser.add_argument('--val_1_path', type=str, default='./data/val_1_no_missings.csv')
    parser.add_argument('--val_2_path', type=str, default='./data/val_2_no_missings.csv')
    parser.add_argument('--train_path_2000', type=str, default='./data/train_no_missings_2000_2.csv')
    parser.add_argument('--val_1_path_2000', type=str, default='./data/val_1_no_missings_2000_2.csv')
    parser.add_argument('--val_2_path_2000', type=str, default='./data/val_2_no_missings_2000_2.csv')
    # Dataset augmentation
    # parser.add_argument('--train_meta_path_no_missing', type=str, default='./Pro/new_train_meta_subs_no_missing_1.csv')
    # parser.add_argument('--val_1_meta_path_no_missing', type=str, default='./Pro/new_val_1_meta_subs_no_missing.csv')
    # parser.add_argument('--val_2_meta_path_no_missing', type=str, default='./Pro/new_val_2_meta_subs_no_missing.csv')
    parser.add_argument('--modality', type=str, default='audio_video_text',
                        choices=['audio', 'video', 'text', 'audio_video', 'audio_video_text'],
                        help='modality to use. if audio_video both audio and video are used')
    parser.add_argument('--video_feature_name', type=str, default='i3d')
    parser.add_argument('--audio_feature_name', type=str, default='vggish')
    parser.add_argument('--text_feature_name', type=str, default='glove')
    parser.add_argument('--video_features_path', type=str, default='./data/i3d_25fps_stack64step64_2stream_npy/')
    parser.add_argument('--audio_features_path', type=str, default='./data/vggish_npy/')
    parser.add_argument('--align_text_features_path', type=str, default='./data/align_text_npy/')

    parser.add_argument('--reference_paths', type=str, nargs='+',
                        default=['./data/val_1_no_missings.json', './data/val_2_no_missings.json'],
                        help='reference paths for 1-by-1 validation')
    parser.add_argument('--reference_paths_2000', type=str, nargs='+',
                        default=['./data/val_1_no_missings_2000.json', './data/val_2_no_missings_2000.json'],
                        help='reference paths for 1-by-1 validation')

    parser.add_argument('--d_vid', type=int, default=1024, help='raw feature dimension')
    parser.add_argument('--d_aud', type=int, default=128, help='raw feature dimension')
    parser.add_argument('--d_text', type=int, default=300, help='raw feature dimension')
    parser.add_argument('--scale_audio', type=int, default=3, help='audio unsample scale')
    parser.add_argument('--scale_video', type=int, default=8, help='video unsample scale')
    parser.add_argument('--word_emb_caps', default='glove.840B.300d', type=str,
                        help='Embedding code name from torchtext.vocab.Vocab')
    parser.add_argument('--unfreeze_word_emb', dest='unfreeze_word_emb', action='store_true',
                        default=False, help='Whether to finetune the pre-trained text embeddings')
    parser.add_argument('--feature_timespan_in_fps', type=int, default=64,
                        help='how many fps the input features will temporally cover')
    parser.add_argument('--fps_at_extraction', type=int, default=25,
                        help='how many fps were used at feature extraction')
    parser.add_argument('--audio_feature_timespan', type=float,
                        default=0.96, help='audio feature timespan')

    ## TRAINING
    parser.add_argument('--procedure', type=str, required=False,
                        choices=['train_cap', 'train_prop', 'evaluate'])
    parser.add_argument('--device_ids', type=int, nargs='+', default=[0], help='separated by a whitespace')
    parser.add_argument('--start_token', type=str, default='<s>', help='starting token')
    parser.add_argument('--end_token', type=str, default='</s>', help='ending token')
    parser.add_argument('--pad_token', type=str, default='<blank>', help='padding token')
    parser.add_argument('--max_len', type=int, default=30, help='maximum size of 1by1 prediction')
    parser.add_argument('--min_freq_caps', type=int, default=1,
                        help='a word should appear min_freq times in train dataset to be in the vocab')
    parser.add_argument('--lr', type=float, default=5e-5, help='lr (if scheduler is constant)')
    parser.add_argument('--milestones', default=[10,25], help='learning rate decay epoch')
    parser.add_argument('--gamma', default=0.5, help='learning rate ratio')
    parser.add_argument('--B', type=int, default=2, help='batch size per device')
    parser.add_argument('--inf_B_coeff', type=int, default=1,
                        help='The batch size on inference will be inf_B_coeff times B arg')
    parser.add_argument('--epoch_num', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--one_by_one_starts_at', type=int, default=1,
                        help='# of epochs to skip before starting 1-by-1 validation (saves time)')
    parser.add_argument('--early_stop_after', type=int, default=30,
                        help='number of epochs to wait for best metric to change before stopping')
    parser.add_argument(
        '--smoothing', type=float, default=0.7,
        help='smoothing coeff (= 0 cross ent loss, more -- stronger smoothing) must be in [0, 1]'
    )
    parser.add_argument('--grad_clip', type=float, default=20, help='max grad norm for gradients')
    parser.add_argument('--pretrained_prop_model_path', type=str,
                        help='path to pre-trained prop model .pt')
    parser.add_argument('--finetune_prop_encoder', dest='finetune_prop_encoder',
                        action='store_true', default=False)
    parser.add_argument('--pretrained_cap_model_path', type=str,
                        help='path to pre-trained cap model .pt')
    parser.add_argument('--finetune_cap_encoder', dest='finetune_cap_encoder',
                        action='store_true', default=False)
    parser.add_argument('--pretrained_bmt_model_path', type=str,
                        help='path to pre-trained cap model .pt')
    parser.add_argument('--finetune_bmt_encoder', dest='finetune_bmt_encoder',
                        action='store_true', default=False)
    parser.add_argument('--inherit_cap_model_path', type=str,
                        help='path to pre-trained cap model .pt')

    parser.add_argument('--obj_coeff', type=float, default=1, help='objectness coeff in loss')
    parser.add_argument('--noobj_coeff', type=float, default=100, help='noobjectness coeff in loss')
    parser.add_argument('--reg_coeff', type=float, default=1, help='regression coeff in loss')
    parser.add_argument('--cen_coeff', type=float, default=1,help='centerness coeff in loss')

    parser.add_argument('--pad_audio_feats_up_to', type=int, default=800,
                        help='max feature length to pad other features to')
    parser.add_argument('--pad_video_feats_up_to', type=int, default=300,
                        help='max feature length to pad other features to')
    parser.add_argument('--pad_text_feats_up_to', type=int, default=2400,
                        help='max faeture length to pad other faetures to')
    parser.add_argument('--nms_tiou_thresh', type=float, help='non-max suppression objectness thr')
    parser.add_argument('--log_dir', type=str, default='./log/')

    ## EVALUATION
    parser.add_argument('--prop_pred_path', type=str, help='path to a .json file with prop preds')
    parser.add_argument('--avail_mp4_path', type=str, default='./data/available_mp4.txt',
                        help='list of available videos')
    parser.add_argument('--tIoUs', type=float, default=[0.3, 0.5, 0.7, 0.9], nargs='+',
                        help='thresholds for tIoU to be used for 1-by-1 validation')
    parser.add_argument(
        '--max_prop_per_vid', type=int, default=100,
        help='max number of proposals to take into considetation in 1-by-1 validation'
    )
    parser.add_argument('--val_prop_meta_path', type=str, help='Only used in eval_on_learnd_props')

    ## MODEL
    parser.add_argument('--is_scale', type=int, default=0, help='using conv to scale sequential dim')
    parser.add_argument('--AV_fusion_mode', type=str, default='add', help='encoder model feature fusion mode')
    parser.add_argument('--AVT_fusion_mode', type=str, default='cat', help='encoder model feature fusion mode')
    parser.add_argument('--strides_fcos', type=int, nargs='+', default=[2, 4, 8, 16, 32, 64, 128, 256],
                        help='stride relative to the original feature map')
    parser.add_argument('--fpn_feature_sizes', type=int, nargs='+', default=[1200, 600, 300, 150, 75, 38, 19, 10],
                        help='every layer position points number')
    parser.add_argument('--C3_inplanes', type=int, default=512, help='layer 3 channel number')
    parser.add_argument('--C4_inplanes', type=int, default=1024, help='layer 4 channel number')
    parser.add_argument('--C5_inplanes', type=int, default=2048, help='layer 5 channel number')
    parser.add_argument('--planes', type=int, default=256, help='modility feature dimensions')
    parser.add_argument('--model', type=str, default='av_transformer',
                        choices=['transformer', 'av_transformer'], help='caption model type')
    parser.add_argument('--dout_p', type=float, default=0.1, help='dropout probability: in [0, 1]')
    parser.add_argument('--dout_p_fcos', type=float, default=0.3, help='dropout probability: in [0, 1]')
    parser.add_argument('--N', type=int, default=2, help='number of layers in a model')
    parser.add_argument('--d_trans_AV', type=int, default=1024, help='AV feature dim after AV fusion')
    parser.add_argument('--d_trans_AVT', type=int, default=512, help='AVT feature dim after AVT fusion')
    parser.add_argument(
        '--d_model', type=int, default=1024,
        help='the internal space in the mullti-headed attention (when input dims of Q, K, V differ)')
    parser.add_argument(
        '--d_model_caps', type=int, default=300,
        help='hidden size of the crossmodal decoder (caption tokens are mapped into this dim)'
    )
    parser.add_argument('--timespan_fcos', type=float, default=0.32, help='duration of a piece of data')
    parser.add_argument('--H', type=int, default=4, help='number of heads in multiheaded attention')
    parser.add_argument(
        '--d_ff_video', type=int, help='size of the internal layer of PositionwiseFeedForward')
    parser.add_argument(
        '--d_ff_audio', type=int, help='size of the internal layer of PositionwiseFeedForward')
    parser.add_argument(
        '--d_ff_text', type=int, help='size of the internal layer of PositionwiseFeedForward')
    parser.add_argument(
        '--d_ff_cap', type=int, help='size of the internal layer of PositionwiseFeedForward')

    parser.add_argument('--layer_norm', dest='layer_norm', action='store_true', default=True,
                        help='whether to use layer norm in proposal generation heads')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='Save th training model path')

    ## DEBUGGING
    parser.add_argument('--keep_train', type=int, default=0, help='Model keep training')
    parser.add_argument('--debug', dest='debug', action='store_true', default=False,
                        help='runs test() instead of main()')
    parser.add_argument('--dont_log', dest='to_log', action='store_false',
                        help='Prevent logging in the experiment.')

    parser.set_defaults(to_log=True)

    args = parser.parse_args()
    pprint(vars(args))
    cfg = Config(args)

    if args.debug:
        # load your test to debug something using the same config as main() would
        # from tests import test_features_max_length
        # test_features_max_length(cfg)
        pass
    else:
        main(cfg)
