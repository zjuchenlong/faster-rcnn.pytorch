import _init_caffe_paths
from caffe_utils import load_caffe, CaffeParamProvider
import caffe
from collections import OrderedDict
import cPickle as pkl

SAVE_PATH = 'bottom_up_36.pkl'
CAFFE_MODEL = _init_caffe_paths.CAFFE_MODEL
CAFFE_PROTOTXT = _init_caffe_paths.CAFFE_PROTOTXT

caffe_net = load_caffe(CAFFE_PROTOTXT, CAFFE_MODEL)
caffe_keys = caffe_net.params.keys()
print('Caffe net has {} layers'.format(len(caffe_keys)))

param_provider = CaffeParamProvider(caffe_net)

# save_params = {}
save_params = OrderedDict()
for each_key in caffe_keys:
    save_params[each_key] = {}
    if each_key == 'conv1':
        save_params[each_key]['kernel'] = param_provider.conv_kernel(each_key)
        save_params[each_key]['biases'] = param_provider.conv_biases(each_key)
    elif 'bn' in each_key:
        save_params[each_key]['mean'] = param_provider.bn_mean(each_key)
        save_params[each_key]['variance'] = param_provider.bn_variance(each_key)
    elif 'scale' in each_key:
        save_params[each_key]['weights'] = param_provider.scale_weights(each_key)
        save_params[each_key]['biases'] = param_provider.scale_biases(each_key)
    elif 'res' in each_key or 'branch' in each_key:
        # all conv layer
        save_params[each_key]['kernel'] = param_provider.conv_kernel(each_key)
        save_params[each_key]['biases'] = param_provider.conv_biases(each_key)
    elif each_key in ['rpn_conv/3x3', 'rpn_cls_score', 'rpn_bbox_pred']:
        save_params[each_key]['kernel'] = param_provider.conv_kernel(each_key)
        save_params[each_key]['biases'] = param_provider.conv_biases(each_key)
    elif each_key in ['cls_score', 'bbox_pred', 'cls_embedding', 'fc_attr', 'attr_score']:
        save_params[each_key]['weights'] = param_provider.fc_weights(each_key)
        save_params[each_key]['biases'] = param_provider.fc_biases(each_key)
    else:
        print('Skip {} keys'.format(each_key))
        import pdb; pdb.set_trace()

print('Save params dict includes {} keys'.format(len(save_params.keys())))
assert set(save_params.keys()) == set(caffe_keys)

with open(SAVE_PATH, 'wb') as f:
    pkl.dump(save_params, f)

print('Done!')