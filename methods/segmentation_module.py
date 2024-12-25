import torch
import torch.nn as nn
import torch.nn.functional as functional

import inplace_abn
from inplace_abn import InPlaceABNSync, InPlaceABN, ABN
from modules.custom_bn import AIN, RandABN, RandInPlaceABNSync, ABR, InPlaceABR, InPlaceABR_R
from functools import partial

import models
from modules import DeeplabV3, DeeplabV2

from torch.nn import functional as F

from .utils import printCUDA

def make_model(opts, cls=None, head_channels=None):
    if opts.norm_act == 'iabn_sync':
        norm = partial(InPlaceABNSync, activation="leaky_relu", activation_param=.01)
    elif opts.norm_act == 'iabn':
        norm = partial(InPlaceABN, activation="leaky_relu", activation_param=.01)
    elif opts.norm_act == 'abn':
        norm = partial(ABN, activation="leaky_relu", activation_param=.01)
    elif opts.norm_act == 'abr':
        norm = partial(ABR, activation="leaky_relu", activation_param=.01)
    elif opts.norm_act == 'iabr':
        norm = partial(InPlaceABR, activation="leaky_relu", activation_param=.01)
    elif opts.norm_act == 'ain':
        norm = partial(AIN, activation="leaky_relu", activation_param=.01)
    elif opts.norm_act == 'rabn':
        norm = partial(RandABN, activation="leaky_relu", activation_param=.01)
    elif opts.norm_act == 'riabn_sync' or opts.norm_act == 'riabn_sync2':
        norm = partial(RandInPlaceABNSync, activation="leaky_relu", activation_param=.01)
    else:
        raise NotImplementedError

    body = models.__dict__[f'net_{opts.backbone}'](norm_act=norm, output_stride=opts.output_stride, 
                                                   mixup_hidden=opts.mixup_hidden, mixup_alpha=opts.mixup_alpha, layer_mix=opts.layer_mix)
    if not opts.no_pretrained:
        pretrained_path = f'pretrained/{opts.backbone}_iabn_sync.pth.tar'  # Use always iabn_sync model
        pre_dict = torch.load(pretrained_path, map_location='cpu')

        new_state = {}
        for k, v in pre_dict['state_dict'].items():
            if "module" in k:
                new_state[k[7:]] = v
            else:
                new_state[k] = v

        if 'classifier.fc.weight' in new_state:
            del new_state['classifier.fc.weight']
            del new_state['classifier.fc.bias']

        body.load_state_dict(new_state)
        del pre_dict  # free memory
        del new_state

    if cls is None:
        if head_channels is None:
            raise ValueError("One among cls and head_channels must be specified.")
        cls = nn.Conv2d(head_channels, opts.num_classes, 1)
        cls.channels = head_channels
    else:
        head_channels = cls.channels

    if opts.deeplab == 'v3':
        head = DeeplabV3(body.out_channels, head_channels, 256, norm_act=norm,
                         out_stride=opts.output_stride, pooling_size=opts.pooling,
                         pooling=not opts.no_pooling, last_relu=opts.relu)
    elif opts.deeplab == 'v2':
        head = DeeplabV2(body.out_channels, head_channels, norm_act=norm,
                         out_stride=opts.output_stride, last_relu=opts.relu)
    else:
        head = nn.Conv2d(body.out_channels, head_channels, kernel_size=1)

    model = SegmentationModule(body, head, head_channels, cls)

    return model


class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256, proj='convmlp', bn_type='torchsyncbn'):
        super(ProjectionHead, self).__init__()

        if proj == 'linear':
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=1),
                nn.SyncBatchNorm(dim_in),
                nn.ReLU(),
                nn.Conv2d(dim_in, proj_dim, kernel_size=1)
            )

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)


class SegmentationModule(nn.Module):

    def __init__(self, body, head, head_channels, classifier):
        super(SegmentationModule, self).__init__()
        self.body = body
        self.head = head
        self.head_channels = head_channels
        self.proj_head = ProjectionHead(dim_in=head_channels, proj_dim=head_channels)
        self.cls = classifier

    def forward(self, x, target=None, lam=None, use_classifier=True, return_feat=False, return_body=False, return_embedding=False,
                only_classifier=False, only_head=False):
        
        # print(self.cls.cls[0].weight.data)

        if only_classifier:
            return self.cls(x)
        elif only_head:
            return self.cls(self.head(x))
        else:
            x_b, y_a, y_b, lam = self.body(x, target=target, lam=lam)
            if isinstance(x_b, dict):
                x_b = x_b["out"]
            out = self.head(x_b)

            out_size = x.shape[-2:]

            if use_classifier:
                sem_logits = self.cls(out)
                sem_logits = functional.interpolate(sem_logits, size=out_size, mode="bilinear", align_corners=False)
            else:
                sem_logits = out

            if return_feat:
                if return_body:
                    if return_embedding:
                        embedding = self.proj_head(out)
                        if target is not None:
                            return sem_logits, out, x_b, embedding, y_a, y_b, lam
                        return sem_logits, out, x_b, embedding
                    return sem_logits, out, x_b
                return sem_logits, out

            return sem_logits

    def freeze(self):
        for par in self.parameters():
            par.requires_grad = False

    def fix_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, inplace_abn.ABN):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def bn_set_momentum(self, momentum=0.0):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, ABN) or isinstance(m, AIN) or isinstance(m, ABR):
                m.momentum = momentum
