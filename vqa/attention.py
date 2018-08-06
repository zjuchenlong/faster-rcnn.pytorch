import math
import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from fc import FCNet
# from bin_mask import BinaryMask
from utils import tensor_l2_norm
from torch.nn.parameter import Parameter

def conv_weight_norm(conv_kernel, conv_norm=0):
    if conv_norm:
        return weight_norm(conv_kernel, dim=None)
    else:
        return conv_kernel

class SigSoftAttention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, output_channel=36, kernel_size=1, stride=1,  \
                        instance_norm=0, padding_type='same', l2_norm=0, concat=1, leaky_relu=None, last_no_relu=None, \
                        num_conv_layer=1, conv_norm=0, softmax=0, dropout=0.2):
        super(SigSoftAttention, self).__init__()

        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding_type = padding_type
        self.l2_norm = l2_norm
        self.concat = concat
        self.softmax = softmax
        self.num_conv_layer = num_conv_layer
        if self.num_conv_layer == 2:
            conv_hid_dim = 256

        self.v_proj = FCNet([v_dim, num_hid], leaky_relu, last_no_relu)
        self.q_proj = FCNet([q_dim, num_hid], leaky_relu, last_no_relu)
        self.dropout = nn.Dropout(dropout)

        assert stride==1
        if padding_type == 'same':
            padding_num = (self.kernel_size-1)/2
        elif padding_type == 'valid':
            padding_num = 0
            self.zero_padding = nn.ConstantPad2d((self.kernel_size-1)/2, -1) # there is a (x+1)/2 operation later
        else:
            raise ValueError

        if self.concat:
            if self.num_conv_layer == 2:
                self.conv2 = conv_weight_norm(nn.Conv2d(2*num_hid, conv_hid_dim, self.kernel_size, self.stride, padding=padding_num, bias=True), conv_norm=conv_norm)
                self.conv2_relu = nn.LeakyReLU(negative_slope=0.3)
                self.conv1 = conv_weight_norm(nn.Conv2d(conv_hid_dim, self.output_channel, self.kernel_size, self.stride, padding=padding_num, bias=True), conv_norm=conv_norm)
            else:
                self.conv1 = conv_weight_norm(nn.Conv2d(2*num_hid, self.output_channel, self.kernel_size, self.stride, padding=padding_num, bias=True), conv_norm=conv_norm)
        else:
            if self.num_conv_layer == 2:
                self.conv2 = conv_weight_norm(nn.Conv2d(num_hid, conv_hid_dim, self.kernel_size, self.stride, padding=padding_num, bias=True), conv_norm=conv_norm)
                self.conv2_relu = nn.LeakyReLU(negative_slope=0.3)
                self.conv1 = conv_weight_norm(nn.Conv2d(conv_hid_dim, self.output_channel, self.kernel_size, self.stride, padding=padding_num, bias=True), conv_norm=conv_norm)
            else:
                self.conv1 = conv_weight_norm(nn.Conv2d(num_hid, self.output_channel, self.kernel_size, self.stride, padding=padding_num, bias=True), conv_norm=conv_norm)

        self.sigmoid = nn.Sigmoid()
        self.instance_norm = instance_norm
        if self.instance_norm:
            if self.num_conv_layer == 2:
                self.conv2_in = nn.InstanceNorm2d(conv_hid_dim)
            self.conv1_in = nn.InstanceNorm2d(self.output_channel) 

    def forward(self, v, q):
        logits = self.logits(v, q)

        if self.softmax:
            w = nn.functional.softmax(logits, 1)
        else:
            w = self.sigmoid(logits)
        #
        # w = self.binarymask(logits)
        # w = (w + 1)/2.0 # [-1, 1] -> [0, 1]
        ## debug
        # w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        batch, k, _ = v.size()
        v_proj = self.v_proj(v) # [batch, k, qdim]
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1)
        num_hid = v_proj.size(2)

        if self.concat:
            if self.l2_norm:
                norm_v_proj = tensor_l2_norm(v_proj)
                norm_q_proj = tensor_l2_norm(q_proj)
                joint_repr  = torch.cat((norm_v_proj, norm_q_proj), 2)
            else:
                joint_repr = torch.cat((v_proj, q_proj), 2)
            joint_repr = joint_repr.transpose(2, 1).reshape(-1, 2*num_hid, 7, 7)
        else:
            joint_repr = v_proj * q_proj
            joint_repr = joint_repr.transpose(2, 1).reshape(-1, num_hid, 7, 7)

        joint_repr = self.dropout(joint_repr)

        if self.num_conv_layer == 2:
            joint_repr = self.conv2(joint_repr)
            if self.instance_norm:
                joint_repr = self.conv2_in(joint_repr)
            joint_repr = self.conv2_relu(joint_repr)

        joint_repr = self.conv1(joint_repr)

        if self.instance_norm:
            joint_repr = self.conv1_in(joint_repr)
        if self.padding_type == 'valid':
            joint_repr = self.zero_padding(joint_repr)
        joint_repr = joint_repr.reshape(-1, self.output_channel, 7*7).transpose(2, 1)

        return joint_repr

class BinSoftAttention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, output_channel=36, kernel_size=1, stride=1,  \
                        instance_norm=0, padding_type='same', l2_norm=0, concat=1, leaky_relu=None, last_no_relu=None, \
                        num_conv_layer=1, conv_norm=0, dropout=0.2):
        super(BinSoftAttention, self).__init__()

        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding_type = padding_type
        self.l2_norm = l2_norm
        self.concat = concat
        self.num_conv_layer = num_conv_layer
        if self.num_conv_layer == 2:
            conv_hid_dim = 256

        self.v_proj = FCNet([v_dim, num_hid], leaky_relu, last_no_relu)
        self.q_proj = FCNet([q_dim, num_hid], leaky_relu, last_no_relu)
        self.dropout = nn.Dropout(dropout)
        # self.linear = weight_norm(nn.Linear(q_dim, 1), dim=None)
        # add convolutional kernel
        assert stride==1
        if padding_type == 'same':
            padding_num = (self.kernel_size-1)/2
        elif padding_type == 'valid':
            padding_num = 0
            self.zero_padding = nn.ConstantPad2d((self.kernel_size-1)/2, -1) # there is a (x+1)/2 operation later
        else:
            raise ValueError 

        if self.concat:
            if self.num_conv_layer == 2:
                self.conv2 = conv_weight_norm(nn.Conv2d(2*num_hid, conv_hid_dim, self.kernel_size, self.stride, padding=padding_num, bias=True), conv_norm=conv_norm)
                self.conv2_relu = nn.LeakyReLU(negative_slope=0.3)
                self.conv1 = conv_weight_norm(nn.Conv2d(conv_hid_dim, self.output_channel, self.kernel_size, self.stride, padding=padding_num, bias=True), conv_norm=conv_norm)
            else:
                self.conv1 = conv_weight_norm(nn.Conv2d(2*num_hid, self.output_channel, self.kernel_size, self.stride, padding=padding_num, bias=True), conv_norm=conv_norm)
        else:
            if self.num_conv_layer == 2:
                self.conv2 = conv_weight_norm(nn.Conv2d(num_hid, conv_hid_dim, self.kernel_size, self.stride, padding=padding_num, bias=True), conv_norm=conv_norm)
                self.conv2_relu = nn.LeakyReLU(negative_slope=0.3)
                self.conv1 = conv_weight_norm(nn.Conv2d(conv_hid_dim, self.output_channel, self.kernel_size, self.stride, padding=padding_num, bias=True), conv_norm=conv_norm)
            else:
                self.conv1 = conv_weight_norm(nn.Conv2d(num_hid, self.output_channel, self.kernel_size, self.stride, padding=padding_num, bias=True), conv_norm=conv_norm)
        self.binarymask = BinaryMask.apply
        self.sigmoid = nn.Sigmoid()
        self.instance_norm = instance_norm
        if self.instance_norm:
            if self.num_conv_layer == 2:
                self.conv2_in = nn.InstanceNorm2d(conv_hid_dim)
            self.conv1_in = nn.InstanceNorm2d(self.output_channel) 

    def forward(self, v, q):
        logits = self.logits(v, q)
        logits = self.sigmoid(logits) - 0.5
        # w = nn.functional.softmax(logits, 1)
        w = self.binarymask(logits)
        w = (w + 1)/2.0 # [-1, 1] -> [0, 1]
        return w

    def logits(self, v, q):
        batch, k, _ = v.size()
        v_proj = self.v_proj(v) # [batch, k, qdim]
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1)
        num_hid = v_proj.size(2)

        if self.concat:
            if self.l2_norm:
                norm_v_proj = tensor_l2_norm(v_proj)
                norm_q_proj = tensor_l2_norm(q_proj)
                joint_repr  = torch.cat((norm_v_proj, norm_q_proj), 2)
            else:
                joint_repr = torch.cat((v_proj, q_proj), 2)
            joint_repr = joint_repr.transpose(2, 1).reshape(-1, 2*num_hid, 7, 7)
        else:
            joint_repr = v_proj * q_proj
            joint_repr = joint_repr.transpose(2, 1).reshape(-1, num_hid, 7, 7)

        joint_repr = self.dropout(joint_repr)

        if self.num_conv_layer == 2:
            joint_repr = self.conv2(joint_repr)
            if self.instance_norm:
                joint_repr = self.conv2_in(joint_repr)
            joint_repr = self.conv2_relu(joint_repr)

        joint_repr = self.conv1(joint_repr)
        
        if self.instance_norm:
            joint_repr = self.conv1_in(joint_repr)
        if self.padding_type == 'valid':
            joint_repr = self.zero_padding(joint_repr)
        joint_repr = joint_repr.reshape(-1, self.output_channel, 7*7).transpose(2, 1)

        return joint_repr

class GridAttention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, leaky_relu=None, last_no_relu=None, dropout=0.2):
        super(GridAttention, self).__init__()

        self.v_proj = FCNet([v_dim, num_hid], leaky_relu, last_no_relu)
        self.q_proj = FCNet([q_dim, num_hid], leaky_relu, last_no_relu)
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(q_dim, 1), dim=None)

    def forward(self, v, q):
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        batch, k, _ = v.size()
        v_proj = self.v_proj(v) # [batch, k, qdim]
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1)
        joint_repr = v_proj * q_proj
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        return logits

# class Attention(nn.Module):
#     def __init__(self, v_dim, q_dim, num_hid, leaky_relu=None, last_no_relu=None,):
#         super(Attention, self).__init__()
#         self.nonlinear = FCNet([v_dim + q_dim, num_hid], leaky_relu, last_no_relu)
#         self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)

#     def forward(self, v, q):
#         logits = self.logits(v, q)
#         w = nn.functional.softmax(logits, 1)
#         return w

#     def logits(self, v, q):
#         num_objs = v.size(1)
#         q = q.unsqueeze(1).repeat(1, num_objs, 1)
#         vq = torch.cat((v, q), 2)
#         joint_repr = self.nonlinear(vq)
#         logits = self.linear(joint_repr)
#         return logits


class SoftAttention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):
        super(SoftAttention, self).__init__()

        self.v_proj = FCNet([v_dim, num_hid])
        self.q_proj = FCNet([q_dim, num_hid])
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(q_dim, 1), dim=None)

    def forward(self, v, q):
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        batch, k, _ = v.size()
        v_proj = self.v_proj(v) # [batch, k, qdim]
        q_proj = self.q_proj(q).unsqueeze(1)
        joint_repr = v_proj * q_proj
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        return logits