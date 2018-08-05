import os
import caffe

def load_caffe(proto, weight, gpu=True):
    if gpu:
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    net = caffe.Net(proto, weight, caffe.TEST)
    return net

class CaffeParamProvider():
    def __init__(self, caffe_net):
        self.caffe_net = caffe_net

    def conv_kernel(self, name):
        k = self.caffe_net.params[name][0].data
        return k

    def conv_biases(self, name):
        if len(self.caffe_net.params[name]) == 1:
            k = None
        else:
            k = self.caffe_net.params[name][1].data
        return k

    def bn_gamma(self, name):
        return self.caffe_net.params[name][0].data

    def bn_beta(self, name):
        return self.caffe_net.params[name][1].data

    def bn_mean(self, name):
        return (self.caffe_net.params[name][0].data/self.caffe_net.params[name][2].data)

    def bn_variance(self, name):
        return (self.caffe_net.params[name][1].data/self.caffe_net.params[name][2].data)

    def scale_weights(self, name):
        w = self.caffe_net.params[name][0].data
        return w

    def scale_biases(self, name):
        b = self.caffe_net.params[name][1].data
        return b

    def fc_weights(self, name):
        w = self.caffe_net.params[name][0].data
        #w = w.transpose((1, 0))
        return w

    def fc_biases(self, name):
        if len(self.caffe_net.params[name]) == 1:
            b = None
        else:
            b = self.caffe_net.params[name][1].data
        return b