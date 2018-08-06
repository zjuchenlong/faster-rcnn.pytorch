import _init_feat_paths
import torch
import pprint
import h5py
import argparse
import numpy as np
from tqdm import tqdm
import os.path as osp
from utils import create_dir
import torch.nn.functional as F
from torch.autograd import Variable
from caffe2th.np2th import load_np_params
# from model.nms.nms_wrapper import nms
from model.nms_bu.nms_wrapper import nms
from model.rpn.bbox_transform import clip_boxes, bbox_transform_inv
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.minibatch import get_minibatch
# from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.faster_rcnn.resnet import resnet

try:
    import cPickle as pkl
except ModuleNotFoundError:
    import _pickle as pkl
try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='coco', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/resnet101_ls.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--np_params_path', dest='np_params_path',
                        help='load the numpy params',
                        default='../../tools/caffe2th/bottom_up_36.pkl', type=str)
    parser.add_argument('--bottom_up_objs_dir', dest='bottom_up_objs_dir',
                        help='botoom_up paper released detected objs',
                        default='../vqa_data/bottom_up_objs', type=str)
    parser.add_argument('--feature_save_dir', dest='feature_save_dir',
                        default='../vqa_data/features/bottom_up_origin')
    # parser.add_argument('--num_detect_boxes', dest='num_detect_boxes',
    #                     help='number of boxes to detect',
    #                     default=36, type=int)
 
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    # from bottom-up feature which trained on vg 1600-400-20; 1601 classes classification task
    num_dummy_classes = 1601 
    dataset_dummy_classes = ['DUMMY']*num_dummy_classes

    args = parse_args()
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    np.random.seed(cfg.RNG_SEED)

    args.dataset == 'coco'
    args.imdb_name = 'coco_2014_train'
    args.imdbval_name = 'coco_2014_val'
    args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5, 1, 2]', \
                    'POOLING_SIZE', '14', 'RESNET.LAYER4_STRIDE', '1', \
                    'RESNET.BRANCH2B_DILATION', 'True', 'RPN.PROPOSAL_TYPE', 'bottom_up', \
                    'TRAIN.MAX_SIZE', '1000']
    args.cfg_file = "../../cfgs/res101_bottomup.yml"

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    cfg.TRAIN.USE_FLPPED = False
    train_imdb, train_roidb, train_ratio_list, train_ratio_index = combined_roidb(args.imdb_name, False)
    val_imdb , val_roidb, val_ratio_list, val_ratio_index = combined_roidb(args.imdbval_name, False)
    print('Finish load imdb and roidb')

    dataset_classes = train_imdb.classes
    assert dataset_classes == val_imdb.classes
    # dataset_classes = val_imdb.classes
    fasterRCNN = resnet(dataset_dummy_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    fasterRCNN.create_architecture()
    print('Finish buid faster architecture')

    # encoding as pkl saved by caffe is from python2.7, now is in python3.6
    with open(args.np_params_path, 'rb') as f:
        np_params = pkl.load(f, encoding='latin1')
    load_np_params(fasterRCNN, np_params)
    # follow bottom-up and top-down
    cfg.POOLING_MODE = 'pool'

    # initilize the tensor holder here.
    im_data_input = torch.FloatTensor(1)
    im_info_input = torch.FloatTensor(1)
    num_boxes_input = torch.LongTensor(1)
    gt_boxes_input = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data_input = im_data_input.cuda()
        im_info_input = im_info_input.cuda()
        num_boxes_input = num_boxes_input.cuda()
        gt_boxes_input = gt_boxes_input.cuda()
        fasterRCNN.cuda()
        cfg.CUDA = True

    # make variable
    im_data_input = Variable(im_data_input, volatile=True)
    im_info_input = Variable(im_info_input, volatile=True)
    num_boxes_input = Variable(num_boxes_input, volatile=True)
    gt_boxes_input = Variable(gt_boxes_input, volatile=True)

    fasterRCNN.eval()
    num_dataset_classes = len(dataset_classes)
    # empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))

    # default set different for w/o vis
    thresh = 0.0

    # for split_name in ['train', 'val']:
    for split_name in ['train']:
    # for split_name in ['val']:
        split_roidb = eval(split_name + '_roidb')
        # max_per_image = 100
        # num_images = len(val_imdb.image_index)
        # all_boxes = [[[] for _ in xrange(num_images)]
        #        for _ in xrange(num_dummy_classes)]
        with open(osp.join(args.bottom_up_objs_dir, '{}_obj.pkl'.format(split_name)), 'rb') as f:
            split_bottom_up_objs = pkl.load(f)

        create_dir(args.feature_save_dir)
        split_h5_path = osp.join(args.feature_save_dir, '{}_feat.h5'.format(split_name))
        split_h5 = h5py.File(split_h5_path, 'w')
        split_id = osp.join(args.feature_save_dir, '{}_id.pkl'.format(split_name))

        split_feat = split_h5.create_dataset('image_features', (len(split_roidb), 36, 2048))

        split_id2idx = {}
        for i, each_item in tqdm(enumerate(split_roidb)):
            blobs = get_minibatch([each_item], num_dataset_classes)

            img_id = each_item['img_id']
            split_id2idx[img_id] = i
            im_data = torch.from_numpy(blobs['data'])
            im_info = torch.from_numpy(blobs['im_info'])
            im_data = im_data.permute(0, 3, 1, 2).contiguous()

            gt_boxes = torch.FloatTensor([1,1,1,1,1])
            num_boxes = torch.LongTensor(np.array([0, ]))

            im_data_input.data.resize_(im_data.size()).copy_(im_data)
            im_info_input.data.resize_(im_info.size()).copy_(im_info)
            gt_boxes_input.data.resize_(gt_boxes.size()).copy_(gt_boxes)
            num_boxes_input.data.resize_(num_boxes.size()).copy_(num_boxes)

            #---------------------------------------------------------------------
            # Faster RCNN
            # ---------------------------------------------------------------------
            # feed image data to base model to obtain base feature map
            base_feat = fasterRCNN.RCNN_base(im_data_input)

            # rois load from pkl file
            saved_roi = split_bottom_up_objs[img_id] * im_info[0][2]
            rois = Variable(torch.from_numpy(np.hstack([np.zeros((36, 1), dtype=np.float32), saved_roi])))
            if args.cuda:
                rois = rois.cuda()
            # feed base feature map tp RPN to obtain rois
            # rois, rpn_loss_cls, rpn_loss_bbox = fasterRCNN.RCNN_rpn(base_feat, im_info_input.data, gt_boxes_input.data, num_boxes_input.data)
            # rois = Variable(rois)
            # do roi pooling based on predicted rois

            if cfg.POOLING_MODE == 'crop':
                # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
                grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
                grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
                pooled_feat = fasterRCNN.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
                if cfg.CROP_RESIZE_WITH_MAX_POOL:
                    pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
            elif cfg.POOLING_MODE == 'align':
                pooled_feat = fasterRCNN.RCNN_roi_align(base_feat, rois.view(-1, 5))
            elif cfg.POOLING_MODE == 'pool':
                pooled_feat = fasterRCNN.RCNN_roi_pool(base_feat, rois.view(-1,5))

            # feed pooled features to top model
            pooled_feat = fasterRCNN._head_to_tail(pooled_feat)

            # # compute bbox offset
            # bbox_pred = fasterRCNN.RCNN_bbox_pred(pooled_feat)
      
            # # compute object classification probability
            # cls_score = fasterRCNN.RCNN_cls_score(pooled_feat)
            # cls_prob = F.softmax(cls_score, dim=1)
            # cls_prob = cls_prob.view(1, rois.size(1), -1) # batch_size is 1
            # bbox_pred = bbox_pred.view(1, rois.size(1), -1)
            # ---------------------------------------------------------------------

            """
            scores = cls_prob.data
            boxes = rois.data[:, :, 1:5]

            if cfg.TEST.BBOX_REG:
                # Apply bounding-box regression deltas
                box_deltas = bbox_pred.data
                # Follow bottom-up Code
                # if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                #     # Optionally normalize targets by a precomputed mean and stdev
                #     if args.class_agnostic:
                #         box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                #                    + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                #         box_deltas = box_deltas.view(1, -1, 4)
                #     else:
                #         box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                #                    + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                #         box_deltas = box_deltas.view(1, -1, 4 * num_dummy_classes)
                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, im_info_input.data, 1)
            else:
                # Simply repeat the boxes, once for each class
                pred_boxes = np.tile(boxes, (1, scores.shape[1]))

            # rescale to origin input size
            pred_boxes /= im_info[0][2]

            scores = scores.squeeze()
            pred_boxes = pred_boxes.squeeze()
            
            # -----------------------------------------------------------------------------------
            # bottom-up and top-down Original: use roi direct do nms
            # -----------------------------------------------------------------------------------
            boxes[0] /= im_info[0][2]
            max_conf = torch.zeros((pred_boxes.shape[0]))
            max_conf_class = torch.zeros((pred_boxes.shape[0])) # to remembe the class of each rois
            if args.cuda:
                max_conf = max_conf.cuda()
                max_conf_class = max_conf_class.cuda()

            for j in xrange(1, num_dummy_classes):
                cls_scores = scores[:, j]
                inds = torch.nonzero(cls_scores > thresh).view(-1)
                if inds.numel() > 0:
                    cls_boxes = boxes[0]
                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                    keep = nms(cls_dets.cpu().numpy(), cfg.TEST.NMS)
                    keep = torch.from_numpy(np.array(keep))
                    if args.cuda:
                        keep = keep.cuda()
                    # keep = nms(cls_dets, cfg.TEST.NMS)
                    # keep = keep.squeeze(1).long()
                    if len(keep[cls_scores[keep] > max_conf[keep]]):
                        max_conf_class[keep[cls_scores[keep] > max_conf[keep]]] = j
                    max_conf[keep] = torch.max(cls_scores[keep], max_conf[keep]) # pytorch 0.3.1 don't have torch.where
                else:
                    pass

            if args.num_detect_boxes > 0 and len(max_conf) > args.num_detect_boxes:
                keep_boxes = torch.sort(max_conf, descending=True)[1][:args.num_detect_boxes]

            detected_result_box = boxes[0][keep_boxes]
            detected_result_feat = pooled_feat[keep_boxes]
            image_id = each_item['img_id']
            print('{}'.format(image_id))
            import pdb; pdb.set_trace()
            print('This scripts is for debug')
            # -----------------------------------------------------------------------------
            """
            split_feat[i] = pooled_feat.data.cpu().numpy()

        with open(split_id, 'wb') as f:
            pkl.dump(split_id2idx, f)

        split_h5.close()
    print('Done!')
