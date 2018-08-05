import torch

def copy_from_a_to_b(params_a, params_b):
    assert params_a.shape == params_b.shape
    if isinstance(params_b, torch.nn.Parameter):
        params_b.data.copy_(torch.from_numpy(params_a))
    elif isinstance(params_b, torch.FloatTensor):
        params_b.copy_(torch.from_numpy(params_a))
    else:
        raise ValueError

def copy_conv(np_dict, np_key, th_net_part):
    copy_from_a_to_b(np_dict[np_key]['kernel'], th_net_part._parameters['weight'])
    if th_net_part._parameters['bias'] is not None:
        copy_from_a_to_b(np_dict[np_key]['biases'], th_net_part._parameters['bias'])
    else:
        assert np_dict[np_key]['biases'] is None
    assert len(th_net_part._buffers) == 0
    del np_dict[np_key]

def copy_fc(np_dict, np_key, th_net_part):
    copy_from_a_to_b(np_dict[np_key]['weights'], th_net_part._parameters['weight'])
    copy_from_a_to_b(np_dict[np_key]['biases'], th_net_part._parameters['bias'])
    del np_dict[np_key]

def copy_bn_scale(np_dict, np_key, th_net_part):
    bn_key = 'bn' + np_key
    scale_key = 'scale' + np_key
    copy_from_a_to_b(np_dict[bn_key]['mean'], th_net_part._buffers['running_mean'])
    copy_from_a_to_b(np_dict[bn_key]['variance'], th_net_part._buffers['running_var'])
    copy_from_a_to_b(np_dict[scale_key]['weights'], th_net_part._parameters['weight'])
    copy_from_a_to_b(np_dict[scale_key]['biases'], th_net_part._parameters['bias'])
    del np_dict[bn_key]
    del np_dict[scale_key]

def copy_bottleneck(np_dict, np_key, th_net_part, downsample=False):
    res_branch2a_key = 'res{}_branch2a'.format(np_key)
    bn_branch2a_key = '{}_branch2a'.format(np_key)
    res_branch2b_key = 'res{}_branch2b'.format(np_key)
    bn_branch2b_key = '{}_branch2b'.format(np_key)
    res_branch2c_key = 'res{}_branch2c'.format(np_key)
    bn_branch2c_key = '{}_branch2c'.format(np_key)
    if downsample:
        res_branch1_key = 'res{}_branch1'.format(np_key)
        bn_branch1_key = '{}_branch1'.format(np_key)
    copy_conv(np_dict, res_branch2a_key, th_net_part.conv1)
    copy_bn_scale(np_dict, bn_branch2a_key, th_net_part.bn1)
    copy_conv(np_dict, res_branch2b_key, th_net_part.conv2)
    copy_bn_scale(np_dict, bn_branch2b_key, th_net_part.bn2)
    copy_conv(np_dict, res_branch2c_key, th_net_part.conv3)
    copy_bn_scale(np_dict, bn_branch2c_key, th_net_part.bn3)
    if downsample:
        copy_conv(np_dict, res_branch1_key, th_net_part.downsample[0])
        copy_bn_scale(np_dict, bn_branch1_key, th_net_part.downsample[1])

def load_np_params(net, params):
    # stage-1
    copy_conv(params, 'conv1', net.RCNN_base[0])
    copy_bn_scale(params, '_conv1', net.RCNN_base[1])
    # stage-2
    copy_bottleneck(params, '2a', net.RCNN_base[4][0], downsample=True)
    copy_bottleneck(params, '2b', net.RCNN_base[4][1])
    copy_bottleneck(params, '2c', net.RCNN_base[4][2])
    # stage-3
    copy_bottleneck(params, '3a', net.RCNN_base[5][0], downsample=True)
    copy_bottleneck(params, '3b1', net.RCNN_base[5][1])
    copy_bottleneck(params, '3b2', net.RCNN_base[5][2])
    copy_bottleneck(params, '3b3', net.RCNN_base[5][3])
    # stage-4
    copy_bottleneck(params, '4a', net.RCNN_base[6][0], downsample=True)
    for i in range(1, 23):
        copy_bottleneck(params, '4b{}'.format(i), net.RCNN_base[6][i])
    # stage-5
    copy_bottleneck(params, '5a', net.RCNN_top[0][0], downsample=True)
    copy_bottleneck(params, '5b', net.RCNN_top[0][1])
    copy_bottleneck(params, '5c', net.RCNN_top[0][2])
    # RPN
    copy_conv(params, 'rpn_conv/3x3', net.RCNN_rpn.RPN_Conv)
    copy_conv(params, 'rpn_cls_score', net.RCNN_rpn.RPN_cls_score)
    copy_conv(params, 'rpn_bbox_pred', net.RCNN_rpn.RPN_bbox_pred)

    copy_fc(params, 'cls_score', net.RCNN_cls_score)
    copy_fc(params, 'bbox_pred', net.RCNN_bbox_pred)

    print('Skip {}'.format(params.keys()))
    print('Finish load numpy feature!')