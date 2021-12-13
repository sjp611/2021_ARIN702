from models.resizer import Resizer
from models.resizer_att import Resizer_att
from models.vgg import vgg16 

import torch
import torch.nn as nn
import torch.nn.functional as F
#from .._internally_replaced_utils import load_state_dict_from_url
from torch.hub import load_state_dict_from_url
from typing import Union, List, Dict, Any, cast



class VGG16_cifar(nn.Module):
    def __init__(self, pretrained = True, num_classes = 10):
        super(VGG16_cifar, self).__init__()
        self.model_func = vgg16(pretrained = pretrained)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x_resize = x
        x = self.model_func(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x, x_resize



#####################################################################################

class VGG16_resizer_cifar(nn.Module):
    def __init__(self, pretrained = True, num_classes = 10, out_resize=224):
        super(VGG16_resizer_cifar, self).__init__()
        self.resizer = Resizer(in_chs=3, out_size = out_resize, n_filters = 16, n_res_blocks = 1, mode='bilinear')
        self.model_func = vgg16(pretrained = pretrained)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x_resize = self.resizer(x)
        x = self.model_func(x_resize)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x, x_resize


#####################################################################################

class VGG16_resizer_att_cifar(nn.Module):
    def __init__(self, pretrained = True, num_classes = 10, out_resize=224):
        super(VGG16_resizer_att_cifar, self).__init__()
        self.resizer = Resizer_att(in_chs=3, out_size = out_resize, n_filters = 16, n_res_blocks = 1, mode='bilinear')

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.model_func = vgg16(pretrained = pretrained)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x_resize = self.resizer(x)
        x = self.model_func(x_resize)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x, x_resize

