import torch
import torch.nn as nn
import torchvision

from torchvision.models.utils import load_state_dict_from_url

from net.mod.batchnorm import SynchronizedBatchNorm2d

BatchNorm2d = SynchronizedBatchNorm2d

__all__ = ['vgg16']

model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
}

ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGGNet(torchvision.models.vgg.VGG):
    def __init__(self, arch, range, layers, requires_grad=True, remove_fc=True):
        super().__init__(layers)
        self.range = range

        if remove_fc:
            del self.classifier
    
    def forward(self, x):
        output = {}

        for idx in range(len(self.range)):
            for layer in range(self.range[idx][0], self.range[idx][1]):
                x = self.features[layer](x)
            output["x%d"%(idx+1)] = x
        
        return output
        

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v , kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, BatchNorm2d(v), nn.ReLU(inpalce=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def _vgg(arch, range, cfg, pretrained, progress, **kwargs):
    model = VGGNet(arch, range, make_layers(cfg))
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch])
        model.load_state_dict(state_dict, strict=False)
    
    return model


def vgg16(pretrained=False, progress=True, **kwargs):
    """VGG16 model

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', ranges['vgg16'], cfgs['D'], pretrained=pretrained, progress=progress, **kwargs)