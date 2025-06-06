from torch import nn
import torch
import pickle5 as pickle
from torch.nn import functional as F
import numpy as np
from torch.nn import Parameter


class IncrementalClassifier(nn.Module):
    def __init__(self, classes, norm_feat=False, channels=256):
        super().__init__()
        self.channels = channels
        self.classes = classes
        self.tot_classes = 0
        for lcl in classes:
            self.tot_classes += lcl
        self.cls = nn.ModuleList(
            [nn.Conv2d(channels, c, 1) for c in classes])
        self.norm_feat = norm_feat

    def forward(self, x):
        if self.norm_feat:
            x = F.normalize(x, p=2, dim=1)
        out = []
        for mod in self.cls:
            out.append(mod(x))
        return torch.cat(out, dim=1)

    def imprint_weights_step(self, features, step):
        self.cls[step].weight.data = features.view_as(self.cls[step].weight.data)

    def imprint_weights_class(self, features, cl, alpha=1):
        step = 0
        while cl >= self.classes[step]:
            cl -= self.classes[step]
            step += 1
        if step == len(self.classes) - 1:  # last step! alpha = 1
            alpha = 0
        self.cls[step].weight.data[cl] = alpha * self.cls[step].weight.data[cl] + \
                                         (1 - alpha) * features.view_as(self.cls[step].weight.data[cl])
        self.cls[step].bias.data[cl] = 0.


class CosineClassifier(nn.Module):
    def __init__(self, classes, channels=256):
        super().__init__()
        self.channels = channels
        self.cls = nn.ModuleList(
            [nn.Conv2d(channels, c, 1, bias=False) for c in classes])
        self.scaler = 10.
        self.classes = classes
        self.tot_classes = 0
        for lcl in classes:
            self.tot_classes += lcl

    def get_centers(self):
        prop_list = []
        for conv in self.cls:
            prop_list.append(conv.weight.squeeze())
        centers = nn.Parameter(torch.concat(prop_list, dim=0).detach().clone())
        return centers

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        out = []
        for i, mod in enumerate(self.cls):
            out.append(self.scaler * F.conv2d(x, F.normalize(mod.weight, dim=1, p=2)))
        return torch.cat(out, dim=1)

    def imprint_weights_step(self, features, step):
        self.cls[step].weight.data = features.view_as(self.cls[step].weight.data)

    def imprint_weights_class(self, features, cl, alpha=1):
        step = 0
        while cl >= self.classes[step]:
            cl -= self.classes[step]
            step += 1
        if step == len(self.classes) - 1:  # last step! alpha = 1
            alpha = 0
        self.cls[step].weight.data[cl] = alpha * self.cls[step].weight.data[cl] + \
                                         (1 - alpha) * features.view_as(self.cls[step].weight.data[cl])


class SPNetClassifier(nn.Module):
    def __init__(self, opts, classes):
        super().__init__()
        datadir = f"data/{opts.dataset}"
        if opts.embedding == 'word2vec':
            class_emb = pickle.load(open(datadir + '/word_vectors/word2vec.pkl', "rb"))
        elif opts.embedding == 'fasttext':
            class_emb = pickle.load(open(datadir + '/word_vectors/fasttext.pkl', "rb"))
        elif opts.embedding == 'fastnvec':
            class_emb = np.concatenate([pickle.load(open(datadir + '/word_vectors/fasttext.pkl', "rb")),
                                        pickle.load(open(datadir + '/word_vectors/word2vec.pkl', "rb"))], axis=1)
        else:
            raise NotImplementedError(f"Embeddings type {opts.embeddings} is not known")

        self.class_emb = class_emb[classes]
        self.class_emb = F.normalize(torch.tensor(self.class_emb), p=2, dim=1)
        self.class_emb = torch.transpose(self.class_emb, 1, 0).float()
        self.class_emb = Parameter(self.class_emb, False)
        self.channels = self.class_emb.shape[0]

    def forward(self, x):
        return torch.matmul(x.permute(0, 2, 3, 1), self.class_emb).permute(0, 3, 1, 2)
