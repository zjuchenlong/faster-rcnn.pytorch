import os.path as osp
import sys

BOTTOM_UP_ATTENTION_PATH = '/home/cl/workspace/baseline/bottom-up-attention'
CAFFE_MODEL = osp.join(BOTTOM_UP_ATTENTION_PATH, \
    'data/faster_rcnn_models/resnet101_faster_rcnn_final_iter_320000.caffemodel')
CAFFE_PROTOTXT = osp.join(BOTTOM_UP_ATTENTION_PATH, \
    'models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt')

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

def assert_exist_path(path):
    assert osp.exists(path)

caffe_path = osp.join(BOTTOM_UP_ATTENTION_PATH, 'caffe/python')
add_path(caffe_path)

bu_lib_path = osp.join(BOTTOM_UP_ATTENTION_PATH, 'lib')
add_path(bu_lib_path)

assert_exist_path(CAFFE_MODEL)
assert_exist_path(CAFFE_PROTOTXT)
