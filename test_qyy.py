
import torch
from models.ATiny_pelee2 import build_net


print("Atinypelee2")
ssd_net = build_net()
#print(ssd_net)

def foo(net, hook=False):  # recursive function
    name_childrens = list(net.named_children())
    if not name_childrens:
        if hook:
            if isinstance(net, torch.nn.Conv2d):
                # net.register_forward_hook(save_hook(net.__class__.__name__))
                # net.register_forward_hook(simple_hook)
                # net.register_forward_hook(simple_hook2)
                print(net)
            if isinstance(net, torch.nn.Linear):
                print(net)
            if isinstance(net, torch.nn.BatchNorm2d):
                print(net)
            if isinstance(net, torch.nn.ReLU):
                print(net)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                print(net)
        return
    for name, module in name_childrens:
        if hook or 'up' in name or 'down' in name:
            foo(module, hook=True)
        else:
            foo(module)

#foo(ssd_net)


input = torch.ones(4, 3, 320, 320)
output = ssd_net(input)
for out in output:
     print(out.size())
