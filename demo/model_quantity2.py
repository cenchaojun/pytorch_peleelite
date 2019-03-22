from __future__ import print_function

import numpy as np
import sys
import os
import torch
import argparse
from load_param import load_param

parser = argparse.ArgumentParser(description='computer cost of designed part modules')

parser.add_argument('-v', '--version', default='ATiny_pelee',
                    help='ATiny_pelee version.')
parser.add_argument('-s', '--size', default='320',
                    help='300,320 or 512 input size.')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO version')

args = parser.parse_args()

dataset = args.dataset
img_dim = int(args.size)
num_classes = (21, 81)[dataset == 'COCO']

#construct network
sys.path.append('../')
if args.version == 'ATiny_pelee':
    from models.ATiny_pelee import build_net
elif args.version == 'SSD_peleenet':
    from models.SSD_peleenet import build_net
elif args.version == 'SSD_ALpelee':
    from models.ALpelee import build_net
elif args.version == 'SSD_ALpelee2':
    from models.ALpelee2 import build_net
elif args.version == 'SSD_ALpelee3':
    from models.ALpelee3 import build_net
elif args.version == 'SSD_ALpelee_little':
    from models.ALpelee_little import build_net
elif args.version == 'Tiny_pelee':
    from models.Tiny_pelee import build_net
else:
    print('Unkown version!  -v error')
    exit()

net = build_net(img_dim, num_classes)
#print(net)

# #load parameter, but needn't do it in fact.
# resume_net_path = os.path.join(
#     '../weights','Atinypelee','SSD_Atinypelee_VOC_320','20190114','SSD_Atinypelee_VOC_epoches_680.pth')
# if not os.path.exists(resume_net_path):
#     print('pth file not exit!')
#     exit()
# load_param(net,resume_net_path)

#compute number of network parameters
num_param = sum(param.nelement() for param in net.parameters())
print('%s %d \nNumber of Parameters :\t%d'%(args.version, img_dim, num_param))

#compute FLOPs and MACC
#reference to ‘https://zhuanlan.zhihu.com/p/33992733’
def print_model_parm_flops(net, img_dim, multiply_adds = False):
    prods = {}
    def save_hook(name):
        def hook_per(self, input, output):
            # print 'flops:{}'.format(self.__class__.__name__)
            # print 'input:{}'.format(input)
            # print '_dim:{}'.format(input[0].dim())
            # print 'input_shape:{}'.format(np.prod(input[0].shape))
            # prods.append(np.prod(input[0].shape))
            prods[name] = np.prod(input[0].shape)
            # prods.append(np.prod(input[0].shape))
        return hook_per

    list_1=[]
    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))
    list_2={}
    def simple_hook2(self, input, output):
        list_2['names'] = np.prod(input[0].shape)


    list_conv=[]
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups) * (2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_conv.append(flops)


    list_linear=[]
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn=[]
    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement())

    list_relu=[]
    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling=[]
    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_pooling.append(flops)


    def foo(net, hook = False):       # recursive function
        name_childrens = list(net.named_children())
        if not name_childrens:
            if hook:
                if isinstance(net, torch.nn.Conv2d):
                    # net.register_forward_hook(save_hook(net.__class__.__name__))
                    # net.register_forward_hook(simple_hook)
                    # net.register_forward_hook(simple_hook2)
                    net.register_forward_hook(conv_hook)
                if isinstance(net, torch.nn.Linear):
                    net.register_forward_hook(linear_hook)
                if isinstance(net, torch.nn.BatchNorm2d):
                    net.register_forward_hook(bn_hook)
                if isinstance(net, torch.nn.ReLU):
                    net.register_forward_hook(relu_hook)
                if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                    net.register_forward_hook(pooling_hook)
                #print(net)
            return
        for name,module in name_childrens:
            #if hook or 'SA'in name or 'CA' in name:
            if hook or 'stem' in name:
                foo(module, hook=True)
            else:
                foo(module)


    foo(net)
    input = torch.rand(3,img_dim,img_dim, requires_grad = True).unsqueeze(0)
    #input = torch.rand(2, 3, img_dim, img_dim, requires_grad=True)  #batch_size = 2
    out = net(input)


    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))

    print(' Number of FLOPs: \t%d ~= %.2fG' % (total_flops,total_flops / 1e9))

print_model_parm_flops(net,img_dim = img_dim)

