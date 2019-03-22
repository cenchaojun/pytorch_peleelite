import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from tensorboardX import SummaryWriter


class Conv_bn_relu(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=1, pad=1, dilation=1, use_relu = True):
        super(Conv_bn_relu, self).__init__()
        self.use_relu = use_relu
        if self.use_relu:
            self.convs = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, pad, dilation = dilation, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, pad, dilation = dilation, bias=False),
                nn.BatchNorm2d(oup,eps=1e-03),
            )

    def forward(self, x):
        out = self.convs(x)
        return out

class ConvDW_bn_relu(nn.Module):
    def __init__(self, inp, kernel_size=3, stride=1, pad=1, use_relu = True):
        super(ConvDW_bn_relu, self).__init__()
        self.use_relu = use_relu
        if self.use_relu:
            self.convs_dw = nn.Sequential(
                    nn.Conv2d(inp, inp, kernel_size, stride, pad, groups=inp, bias=False),
                    nn.BatchNorm2d(inp),
                    nn.ReLU(inplace=True),
            )
        else:
            self.convs_dw = nn.Sequential(
                    nn.Conv2d(inp, inp, kernel_size, stride, pad, groups=inp, bias=False),
                    nn.BatchNorm2d(inp),
            )

    def forward(self, x):
        out = self.convs_dw(x)
        return out


        
class StemBlock(nn.Module):
    def __init__(self, inp=3):
        super(StemBlock, self).__init__()

        self.stem_1 = Conv_bn_relu(inp, 64, 3, 2, 1)

        self.stem_2 = ConvDW_bn_relu(64, 3, 1, 1)

        self.stem_3 = nn.Conv2d(64, 128, 1, 1, 0)

        self.stem_4 = ConvDW_bn_relu(128, 3, 1, 1)

        self.stem_5 = nn.Conv2d(128, 128, 1, 1, 0)

        self.stem_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        stem_1_out  = self.stem_1(x)
        stem_2_out = self.stem_2(stem_1_out)
        stem_3_out = self.stem_3(stem_2_out)
        stem_4_out = self.stem_4(stem_3_out)
        stem_5_out = self.stem_5(stem_4_out)
        out = self.stem_pool(stem_5_out)
        return out


class DenseBlock(nn.Module):
    def __init__(self, inp, inter_channel):
        super(DenseBlock, self).__init__()

        self.cb1_a = Conv_bn_relu(inp,inter_channel, 1, 1, 0, use_relu = False)
        self.cb1_b = Conv_bn_relu(inter_channel, inter_channel, 3, 1, 1)

        self.cb2_a = Conv_bn_relu(inp,inter_channel, 1, 1, 0, use_relu = False)
        self.cb2_b = Conv_bn_relu(inter_channel, inter_channel, 3, 1, 1)
        self.cb2_c = Conv_bn_relu(inter_channel, inter_channel, 3, 1, 2, 2)
        '''

        '''

    def forward(self, x):
        cb1_a_out = self.cb1_a(x)
        cb1_b_out = self.cb1_b(cb1_a_out)

        cb2_a_out = self.cb2_a(x)
        cb2_b_out = self.cb2_b(cb2_a_out)
        cb2_c_out = self.cb2_c(cb2_b_out)

        out = torch.cat((x,cb1_b_out,cb2_c_out),1)
        '''
        cb
        '''
        return out


class TransitionBlock(nn.Module):
    def __init__(self, inp, oup):
        super(TransitionBlock, self).__init__()

        self.tb = Conv_bn_relu(inp,oup,1,1,0)

    def forward(self, x):
        out = self.tb(x)
        return out
                
class Eltwise(nn.Module):
    def __init__(self, operation='+'):
        super(Eltwise, self).__init__()
        self.operation = operation

    def __repr__(self):
        return 'Eltwise %s' % self.operation

    def forward(self, *inputs):
        if self.operation == '+' or self.operation == 'SUM':
            x = inputs[0]
            for i in range(1,len(inputs)):
                x = x + inputs[i]
        elif self.operation == '*' or self.operation == 'MUL':
            x = inputs[0]
            for i in range(1,len(inputs)):
                x = x * inputs[i]
        elif self.operation == '/' or self.operation == 'DIV':
            x = inputs[0]
            for i in range(1,len(inputs)):
                x = x / inputs[i]
        elif self.operation == 'MAX':
            x = inputs[0]
            for i in range(1,len(inputs)):
                x =torch.max(x, inputs[i])
        else:
            print('forward Eltwise, unknown operator')
        return x

class PeleeNet(nn.Module):
    def __init__(self, nDenseBlocks = [3,4,8,6]):
        super(PeleeNet, self).__init__()

        self.stage = nn.Sequential()

        inter_channel = [32,48,64,80]
        total_filter = [128,128,256,64]
        dense_inp = [128,128,128,256]
        with_pooling = [1,1,0,0]

        # building stemblock
        self.stage.add_module('stage_0', StemBlock(3))

        # building middle stageblock
        for i in range(4) :
            self.stage.add_module('stage_{}'.format(i+1),self._make_dense_transition(dense_inp[i], total_filter[i],
                                                                                     inter_channel[i],nDenseBlocks[i],with_pooling[i]))

    def _make_dense_transition(self, dense_inp,total_filter, inter_channel, ndenseblocks,with_pooling = 1):
        layers = []

        for i in range(ndenseblocks):
            layers.append(DenseBlock(dense_inp, inter_channel))
            dense_inp += inter_channel * 2

        #Transition Layer without Compression
        layers.append(TransitionBlock(dense_inp,total_filter))
        if with_pooling == 1:
            layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
            
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.stage(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

'''
if __name__ == '__main__':
    p = PeleeNet()
    #input = torch.ones(1, 3, 320, 320)
    #with SummaryWriter(comment='PeleeNet') as w:
        #w.add_graph(p, (input,))

    #output = p(input)
    from visualize import make_dot
    from torch.autograd import Variable
    x = Variable(torch.randn(8,3,320,320))#change 12 to the channel number of network input
    y = p(x)
    g = make_dot(y)
    g.view()

    #print(output.size())
    #print(p)

    # torch.save(p.state_dict(), 'peleenet.pth.tar')
'''



















