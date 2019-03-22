import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from .Atinypeleenet import *
from tensorboardX import SummaryWriter


class DownSampleLeft(nn.Module):
    def __init__(self, pad = 0):
        super(DownSampleLeft,self).__init__()

        self.Left = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding = pad),
            Conv_bn_relu(128, 64, 1, 1, 0),
        )

    def forward(self, x):
        out = self.Left(x)
        return out

class DownSampleRight(nn.Module):
    def __init__(self):
        super(DownSampleRight,self).__init__()

        self.Right = nn.Sequential(
            nn.Conv2d(128, 64, 1, 1, 0, bias=False),
            ConvDW_bn_relu(64, 3, 2, 1),
        )

    def forward(self, x):
        out = self.Right(x)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, ratio=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1) #target output size of 1x1 
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(128, 128 // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(128 // 16, 128, 1, bias=False)

        self.softmax = nn.Softmax2d()

    def forward(self, x):
        pool_out = self.avg_pool(x) + self.max_pool(x)
        out = self.fc2(self.relu1(self.fc1(pool_out)))
        return self.softmax(out)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)

        self.softmax = nn.Softmax2d()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.softmax(x)


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base PeleeNet network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: PeleeNet layers for input, size of either 750 or 800
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, num_classes, size, base, head):
        super(SSD, self).__init__()
        self.num_classes = num_classes
        self.size = size
                
        # SSD network
        self.peleenet = base
        self.softmax = nn.Softmax()

        self.downSampleLeft1 = DownSampleLeft()
        self.downSampleLeft2 = DownSampleLeft()
        self.downSampleLeft3 = DownSampleLeft()
        if self.size == 512:
            self.downSampleLeft4 = DownSampleLeft()
            self.downSampleLeft5 = DownSampleLeft()
        else:
            self.downSampleLeft4 = DownSampleLeft(pad = 1)
            self.downSampleLeft5 = DownSampleLeft(pad = 1)

        self.downSampleRight2 = DownSampleRight()
        self.downSampleRight3 = DownSampleRight()
        self.downSampleRight4 = DownSampleRight()
        self.downSampleRight5 = DownSampleRight()
        
        self.up_convdw6 = nn.Conv2d(128, 128, 3, 1, 1, groups=128, bias=True)
        self.up_convdw5 = nn.Conv2d(128, 128, 3, 1, 1, groups=128, bias=True)
        self.up_relu5 = nn.ReLU()
        self.up_convdw4 = nn.Conv2d(128, 128, 3, 1, 1, groups=128, bias=True)
        self.up_relu4 = nn.ReLU()
        self.up_convdw3 = nn.Conv2d(128, 128, 3, 1, 1, groups=128, bias=True)
        self.up_relu3 = nn.ReLU()
        self.up_convdw2 = nn.Conv2d(128, 128, 3, 1, 1, groups=128, bias=True)
        
        self.Second_up_conv = nn.Conv2d(128,128,1,1,0)
        self.up_relu2 = nn.ReLU()
        self.up_relu1 = nn.ReLU()

        self.SA1 = SpatialAttention()
        self.SA2 = SpatialAttention()
        self.SA3 = SpatialAttention()
        self.SA4 = SpatialAttention()
        self.SA5 = SpatialAttention()
        self.SA6 = SpatialAttention()

        self.CA1 = ChannelAttention()
        self.CA2 = ChannelAttention()
        self.CA3 = ChannelAttention()
        self.CA4 = ChannelAttention()
        self.CA5 = ChannelAttention()
        self.CA6 = ChannelAttention()

        # detection head        
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

    def forward(self, x, test=False):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()
        First = x
              
        for name,module in next(self.peleenet.children()).named_children():          
            if name == 'stage_2':
                #print(name)
                for ids,module1 in enumerate(module):
                    #print(ids)
                    if ids == 6:
                        #print(ids)
                        x = module1(x)
                        First = x
                        #print(x.device)
                    else:
                      x = module1(x)                      
            else:
              x = module(x)
        
        Second = torch.cat((x,self.downSampleLeft1(First)),1)

        Third = torch.cat((self.downSampleLeft2(Second),self.downSampleRight2(Second)),1)

        Fourth = torch.cat((self.downSampleLeft3(Third),self.downSampleRight3(Third)),1)
        
        Fifth = torch.cat((self.downSampleLeft4(Fourth),self.downSampleRight4(Fourth)),1)
        
        Sixth = torch.cat((self.downSampleLeft5(Fifth),self.downSampleRight5(Fifth)),1)
               
        Fifth_out = self.up_relu5(self.up_convdw6(F.interpolate(Sixth, size = Fifth.size()[2], mode = 'bilinear')) + Fifth)

        Fourth_out = self.up_relu4(self.up_convdw5(F.interpolate(Fifth_out, size = Fourth.size()[2], mode = 'bilinear')) + Fourth)
        
        Third_out = self.up_relu3(self.up_convdw4(F.interpolate(Fourth_out, size = Third.size()[2], mode = 'bilinear')) + Third)
        
        Second_out = self.up_relu2(self.up_convdw3(F.interpolate(Third_out, size = Second.size()[2], mode = 'bilinear')) + Second)
        
        First_out = self.up_relu1(self.Second_up_conv(self.up_convdw2(F.interpolate(Second_out, size = First.size()[2], mode = 'bilinear'))) + First)
        
        First_out = self.SA1(First_out) * self.CA1(First_out) * First_out
        Second_out = self.SA2(Second_out) * self.CA2(Second_out) * Second_out
        Third_out = self.SA3(Third_out) * self.CA3(Third_out) * Third_out
        Fourth_out = self.SA4(Fourth_out) * self.CA4(Fourth_out) * Fourth_out
        Fifth_out = self.SA5(Fifth_out) * self.CA5(Fifth_out) * Fifth_out
        Sixth = self.SA6(Sixth) * self.CA6(Sixth) * Sixth

        First_out_norm = F.normalize(First_out)

        sources.append(First_out_norm)

        Second_out_norm = F.normalize(Second_out)

        sources.append(Second_out_norm)

        Third_out_norm = F.normalize(Third_out)

        sources.append(Third_out_norm)

        Fourth_norm = F.normalize(Fourth)

        sources.append(Fourth_norm)

        Fifth_norm = F.normalize(Fifth)

        sources.append(Fifth_norm)

        Sixth_norm = F.normalize(Sixth)

        sources.append(Sixth_norm)

        # apply multibox head to source layers  here is the Permute layer after loc and conf
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())


        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if test:
            output = (
                loc.view(loc.size(0), -1, 4),                   # loc preds        mbox_loc
                self.softmax(conf.view(-1, self.num_classes)),                # conf preds       mbox_conf
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
            )
        return output

    def init_model(self, base_model_path):

        if os.path.exists(base_model_path):
            base_weights = torch.load(base_model_path)
            print('Loading base network...')
            self.peleenet.layers.load_state_dict(base_weights)

            def xavier(param):
                init.xavier_uniform(param)


            def weights_init(m):
                for key in m.state_dict():
                    if key.split('.')[-1] == 'weight':
                        if 'conv' in key:
                            init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                        if 'bn' in key:
                            m.state_dict()[key][...] = 1
                    elif key.split('.')[-1] == 'bias':
                        m.state_dict()[key][...] = 0
            print('Initializing weights...')
            self.extras.apply(weights_init)
            self.loc.apply(weights_init)
            self.conf.apply(weights_init)

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

def multibox(config, num_classes):
    loc_layers = []
    conf_layers = []
    for k,v in enumerate(config):
        #name_loc = "ext/pm" + str(k+1) + "_mbox_loc"
        loc_layers += [nn.Sequential(
            nn.Conv2d(128, v * 4, kernel_size=1, padding=0),
            ConvDW_bn_relu(v * 4, kernel_size=3, stride=1, pad=1, use_relu = False),
            Conv_bn_relu(v * 4, v * 4, kernel_size=1, stride=1, pad=0, use_relu = False),
        )]
        #name_conf = "ext/pm" + str(k+1) + "_mbox_conf"
        conf_layers += [nn.Sequential(
            nn.Conv2d(128, v * num_classes, kernel_size=1, padding=0),
            ConvDW_bn_relu(v * num_classes, kernel_size=3, stride=1, pad=1, use_relu = False),
            Conv_bn_relu(v * num_classes, v * num_classes, kernel_size=1, stride=1, pad=0, use_relu = False),
        )]
 
    return (loc_layers, conf_layers)   

mbox = [4, 6, 6, 6, 4, 4]

def build_net(size=320, num_classes=21):

    peleenet = PeleeNet()
    head_ = multibox(mbox, num_classes)    
    model = SSD(num_classes, size, peleenet, head_)
    return model
    
'''
if __name__ == '__main__':

    print("Atinypelee")
    ssd_net = build_net()
    #print(ssd_net)
    #input = torch.ones(8, 3, 320, 320)
    
    #output = ssd_net(input)
    #with SummaryWriter('log', comment='TinyPeleeAttention') as w:
        #w.add_graph(ssd_net, (input,))
    
    #for out in output:
        #print(out.size())
    #print(output.size())
    #for key in ssd_net.state_dict().keys():
        #print(key)
        #print(ssd_net.state_dict()[key].size())
    from visualize import make_dot
    x = Variable(torch.randn(8,3,320,320))#change 12 to the channel number of network input
    y = ssd_net(x)
    g = make_dot(y)
    g.view()
    
    #torch.save(ssd_net.state_dict(), 'Atinypelee.pth')
'''