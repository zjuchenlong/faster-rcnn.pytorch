import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset import Dictionary, VQAFeatureDataset
from train import train, evaluate
import utils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='soft')
    parser.add_argument('--feature_dir', type=str, default='bottom_up_origin')
    parser.add_argument('--output', type=str, default='saved_models/bottom_up_origin')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=2018, help='random seed')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--pretrained_model', type=str)
    # parser.add_argument('--image_size', type=int, default=448, help='224|448')
    parser.add_argument('--cpu_size', type=int, default=32, help='32|64|128')
    # parser.add_argument('--output_channel', type=int, default=36)
    # parser.add_argument('--kernel_size', type=int, default=1)
    # parser.add_argument('--stride', type=int, default=1)
    # parser.add_argument('--instance_norm', type=int, default=0, help='wheter to use instance norm in convolution output')
    # parser.add_argument('--padding_type', type=str, default='same', help='same|valid')
    # parser.add_argument('--all_l2_norm', type=int, default=0, help='use l2 norm on visual feature and question feature')
    # parser.add_argument('--l2_norm', type=int, default=0, help='use l2 norm to normalize the feature')
    # parser.add_argument('--concat', type=int, default=1, help='concat two features or not')
    # parser.add_argument('--leaky_relu', type=int, default=0, help='use leaky relu in FC layer')
    # parser.add_argument('--last_no_relu', type=int, default=0, help='the last layer of FC use no relu')
    # parser.add_argument('--num_conv_layer', type=int, default=1, help='number of convolutional layers used')
    # parser.add_argument('--conv_norm', type=int, default=0, help='weight_norm for convolutional kernel')
    # parser.add_argument('--softmax', type=int, default=0, help='use softmax instead of sigmoid')
    # parser.add_argument('--regularizer', type=int, default=0, help='use the diversity regularizer')
    # parser.add_argument('--lambda_diversity', type=float, default=0.1, help='parameter for diversity loss')
    parser.add_argument('--retrain', type=int, default=0, help='reload a pretrained model')
    # parser.add_argument('--optimizer', type=str, default='adamax')
    # parser.add_argument('--lr', type=float, default=0.1, help='for sgd optimizer')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    # assert args.model in ['baseline0_newatt', 'grid', 'grid_box', 'grid_448_box', 'grid_mid_objs', 'grid_mid_grid', 'bin_soft', 'sig_soft']
    assert args.model in ['soft']
    assert args.cpu_size in [32, 64, 128]
    # assert args.padding_type in ['same', 'valid']
    # assert args.num_conv_layer in [1, 2]

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    dictionary = Dictionary.load_from_file('vqa_data/dictionary.pkl')
    train_dset = VQAFeatureDataset('train', dictionary, args)
    eval_dset = VQAFeatureDataset('val', dictionary, args)
    batch_size = args.batch_size

    if args.model == 'soft':
        from soft_attention import build_attention_model
        model = build_attention_model(train_dset, args).cuda()
    else:
        raise ValueError

    # constructor = 'build_%s' % args.model
    # if args.model in ['baseline0_newatt', 'grid_box', 'grid_448_box', 'grid_mid_objs', 'grid_mid_grid']:
    #     constructor = 'build_baseline0_newatt'
    #     import base_model
    #     model = getattr(base_model, constructor)(train_dset, args.num_hid, args.leaky_relu, args.last_no_relu).cuda()
    # elif args.model == 'grid':
    #     import grid_model
    #     model = getattr(grid_model, constructor)(train_dset, args.num_hid, args.all_l2_norm, args.leaky_relu, args.last_no_relu).cuda()
    # elif args.model == 'bin_soft':
    #     import bin_soft_model
    #     model = getattr(bin_soft_model, constructor)(train_dset, args.num_hid, args.output_channel, \
    #             args.kernel_size, args.stride, args.instance_norm, args.padding_type, \
    #             args.l2_norm, args.concat, args.leaky_relu, args.last_no_relu, args.num_conv_layer, args.conv_norm).cuda()
    # elif args.model == 'sig_soft':
    #     import sig_soft_model
    #     model = getattr(sig_soft_model, constructor)(train_dset, args.num_hid, args.output_channel, \
    #             args.kernel_size, args.stride, args.instance_norm, args.padding_type, \
    #             args.l2_norm, args.concat, args.leaky_relu, args.last_no_relu, args.num_conv_layer, \
    #             args.conv_norm, args.softmax).cuda()
    # else:
    #     raise ValueError

    model.w_emb.init_embedding('vqa_data/glove6b_init_300d.npy')
    # model = nn.DataParallel(model).cuda()

    if args.retrain:
        model.load_state_dict(torch.load(args.pretrained_model))

    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=1)
    eval_loader =  DataLoader(eval_dset, batch_size, shuffle=False, num_workers=1)

    if args.mode == 'train':
        train(model, train_loader, eval_loader, args)
    elif args.mode == 'test':
        assert os.path.exists(args.pretrained_model)
        model.load_state_dict(torch.load(args.pretrained_model))
        # model.train(False)
        model.eval()
        eval_score, bound = evaluate(model, eval_loader)

        print('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))
    else:
        raise ValueError