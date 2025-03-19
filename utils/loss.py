import torch.nn as nn
import torch.nn.functional as F
import torch
from abc import ABC

from methods.utils import printCUDA

import torch.nn.functional


class CosineKnowledgeDistillationLoss(nn.Module):
    def __init__(self, reduction='mean', norm='L2'):
        super().__init__()
        self.reduction = reduction
        self.norm = norm.upper()

    def forward(self, inputs, targets):
        inputs = inputs.narrow(1, 0, targets.shape[1])

        if self.norm == "L2":
            loss = ((inputs - targets)**2).mean(dim=1)
        else:
            loss = (inputs - targets).mean(dim=1)

        if self.reduction == 'mean':
            outputs = torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = torch.sum(loss)
        else:
            outputs = loss

        return outputs
    

# class AttentionFeatureDistillationLoss(nn.Module):
#     def __init__(self, alpha=0.5, temperature=2.0):
#         super(AttentionFeatureDistillationLoss, self).__init__()
#         self.alpha = alpha
#         self.temperature = temperature

#     def attention_compute(self, std_features, tch_features):


#     def forward(self, std_features, tch_features, old_classes):



class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, reduction='mean', alpha=1.):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
        # self.mask_encoding = 

    def forward(self, inputs, targets, masks=None):
        inputs = inputs.narrow(1, 0, targets.shape[1])

        if masks is not None:
            masks = masks.unsqueeze(1)
            # inputs = inputs * masks.to(inputs.device)
            # targets = targets * masks.to(inputs.device)
            M = torch.ones_like(masks).to(inputs.device)
            M[masks == 1] = 0.5

        outputs = torch.log_softmax(inputs, dim=1)
        targets = torch.softmax(targets / self.alpha, dim=1)

        loss = -(outputs * targets * M).mean(dim=1) * (self.alpha ** 2)

        if self.reduction == 'mean':
            outputs = torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = torch.sum(loss)
        else:
            outputs = loss

        return outputs


# MiB Losses
class UnbiasedCrossEntropy(nn.Module):
    def __init__(self, old_cl, reduction='mean', ignore_index=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.old_cl = old_cl

    def forward(self, inputs, targets):

        old_cl = self.old_cl
        outputs = torch.zeros_like(inputs)  # B, C (1+V+N), H, W
        den = torch.logsumexp(inputs, dim=1)                               # B, H, W       den of softmax
        outputs[:, 0] = torch.logsumexp(inputs[:, 0:old_cl], dim=1) - den  # B, H, W       p(O)
        outputs[:, old_cl:] = inputs[:, old_cl:] - den.unsqueeze(dim=1)    # B, N, H, W    p(N_i)

        labels = targets.clone()    # B, H, W
        labels[targets < old_cl] = 0  # just to be sure that all labels old belongs to zero

        loss = F.nll_loss(outputs, labels, ignore_index=self.ignore_index, reduction=self.reduction)

        return loss


class UnbiasedKnowledgeDistillationLoss(nn.Module):
    def __init__(self, reduction='mean', alpha=1.):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets, mask=None):

        new_cl = inputs.shape[1] - targets.shape[1]

        targets = targets * self.alpha

        new_bkg_idx = torch.tensor([0] + [x for x in range(targets.shape[1], inputs.shape[1])]).to(inputs.device)

        den = torch.logsumexp(inputs, dim=1)                          # B, H, W
        outputs_no_bgk = inputs[:, 1:-new_cl] - den.unsqueeze(dim=1)  # B, OLD_CL, H, W
        outputs_bkg = torch.logsumexp(torch.index_select(inputs, index=new_bkg_idx, dim=1), dim=1) - den     # B, H, W

        labels = torch.softmax(targets, dim=1)                        # B, BKG + OLD_CL, H, W

        # make the average on the classes 1/n_cl \sum{c=1..n_cl} L_c
        loss = (labels[:, 0] * outputs_bkg + (labels[:, 1:] * outputs_no_bgk).sum(dim=1)) / targets.shape[1]

        if mask is not None:
            loss = loss * mask.float()

        if self.reduction == 'mean':
            outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = -torch.sum(loss)
        else:
            outputs = -loss

        return outputs


class CosineLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.crit = nn.CosineSimilarity(dim=1)

    def forward(self, x, y):
        loss = 1 - self.crit(x, y)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        else:
            loss = loss
        return loss
    

# https://github.com/tfzhou/ContrastiveSeg/blob/main/lib/loss/loss_contrast.py
class PixelContrastLoss(nn.Module, ABC):
    def __init__(self, temperature, base_temperature, ignore_label=[0], max_samples=1024, max_views=100):
        super(PixelContrastLoss, self).__init__()

        self.temperature = temperature
        self.base_temperature = base_temperature

        self.ignore_label = ignore_label

        self.max_samples = max_samples      # 一个batch中所有对比pixel的个数最大值
        self.max_views = max_views          # 一张图片中一个cls的pixel个数最大值

    def _hard_anchor_sampling(self, X, y_hat, y, filenames=None):
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x not in self.ignore_label]
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]   # cls在图片中的占比要至少大于max_view

            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()   # 一个batch中[所有出现类别，每个了类别选中的pixel数]
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]

            for cls_id in this_classes:
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    print('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                    raise Exception

                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                # idx = torch.zeros((128 * 128), dtype=torch.float)
                # idx[hard_indices] = 1
                # idx[easy_indices] = 0.5
                # idx = idx.reshape(128, 128)
                # name = filenames[ii].split('/')[-1].split('.')[0]
                # plt.imsave(f"contrast/{name}_cls{cls_id}.png", idx, cmap='gray')

                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1

        return X_, y_

    def _contrastive(self, feats_, labels_):
        anchor_num, n_view = feats_.shape[0], feats_.shape[1]

        labels_ = labels_.contiguous().view(-1, 1)
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda()

        contrast_count = n_view
        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                     0)
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits + 1e-10)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(self, labels, output, feats, filenames=None):
        # feats = outputs['embedding']     # 32,32
        batch_size = feats.shape[0]

        labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels,
                                                 (feats.shape[2], feats.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()

        _, predict = output.max(dim=1)
        predict = predict.unsqueeze(1).float().clone()
        predict = torch.nn.functional.interpolate(predict,
                                                 (feats.shape[2], feats.shape[3]), mode='nearest')
        predict = predict.squeeze(1).long()

        assert predict.shape[-1] == labels.shape[-1] and labels.shape[-1] == feats.shape[-1], f"contrast: \
            predict - {predict.shape}, labels - {labels.shape}, feats - {feats.shape}"

        # for i in range(batch_size):
        #     name = filenames[i].split('/')[-1].split('.')[0]
        #     plt.imsave(f"./contrast/{name}_gt.png", labels[i].cpu().numpy(), cmap='gray')
        #     plt.imsave(f"./contrast/{name}_pred.png", predict[i].cpu().numpy(), cmap='gray')

        labels = labels.contiguous().view(batch_size, -1)       # B,H*W
        predict = predict.contiguous().view(batch_size, -1)     # B,H*W

        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])    # B,D,H,W -> B,H*W,D

        feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict, filenames)

        loss = self._contrastive(feats_, labels_)
        return loss
    

def MyCrossEntropy(inputs, targets_a, targets_b, lam, num_classes=16, ignore_index=255, reduction='mean'):
    mask_a = (targets_a != ignore_index)
    mask_b = (targets_b != ignore_index)

    targets_a[targets_a == ignore_index] = 0
    targets_b[targets_b == ignore_index] = 0

    targets_a = F.one_hot(targets_a, num_classes=num_classes).permute(0, 3, 1, 2)
    targets_b = F.one_hot(targets_b, num_classes=num_classes).permute(0, 3, 1, 2)
    
    log_probs = F.log_softmax(inputs, dim=1)

    # nll_loss = -(targets * log_probs).sum(dim=1)

    # print(f"target - {targets_a.shape}, log - {log_probs.shape}, mask - {mask_a.shape}")

    nll_loss_a = - (targets_a * log_probs).sum(dim=1) * mask_a
    nll_loss_b = - (targets_b * log_probs).sum(dim=1) * mask_b

    if reduction == 'mean':
        loss = lam * nll_loss_a.sum() / mask_a.sum() + (1 - lam) * nll_loss_b.sum() / mask_b.sum()
    elif reduction == 'sum':
        loss = lam * nll_loss_a.sum() + (1 - lam) * nll_loss_b.sum()
    elif reduction == 'none':
        loss = lam * nll_loss_a + (1 - lam) * nll_loss_b
    
    return loss


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py#L1404


class prototypeContrastLoss(nn.Module):
    def __init__(self, temperature, ignore_label=[]) -> None:
        super().__init__()

        self.temperature = temperature
        self.ignore_label = ignore_label

    def forward(self, features, labels):
        prototypes = {}

        labels = labels.unsqueeze(1).float().clone()    # (B, 1, H, W)
        labels = torch.nn.functional.interpolate(labels,
                                                 (features.shape[2], features.shape[3]), mode='bilinear')
        labels = labels.squeeze(1).long()
        
        for feature, label in zip(features, labels):
            for cls in label.unique():
                if cls in self.ignore_label:
                    continue
                cls = cls.item()
                if cls in prototypes:
                    prototypes[cls].append(F.normalize(torch.einsum('dhw,hw->d', feature, (label==cls).float()), dim=-1))
                else:
                    prototypes[cls] = [F.normalize(torch.einsum('dhw,hw->d', feature, (label==cls).float()), dim=-1)]
        
        loss = 0

        # for cls, prototype in prototypes.items():


        for cls, prototype in prototypes.items():
            pos_prototypes = torch.stack(prototype)
            p_logits = torch.einsum('nd,md->nm', pos_prototypes, pos_prototypes).flatten().unsqueeze(0)
            n_logits = []

            for _cls, _p in prototypes.items():
                neg_prototypes = torch.stack(_p)
                _n_logits = torch.einsum('nd,md->nm', pos_prototypes, neg_prototypes).flatten()
                n_logits.append(_n_logits)
            
            n_logits = torch.concat(n_logits).unsqueeze(0)
            logits = torch.concat([p_logits, n_logits], dim=1) / self.temperature
            lbls = torch.zeros(logits.shape[0], p_logits.shape[1]).to(logits.device)

            loss += F.cross_entropy(logits, lbls, reduction='mean')
        
        return loss


# class GoogleContrastiveLoss(nn.Module):
#     def __init__(self, ignore_labels, num_projection_layers=2, num_projection_channels=256, temperature=0.07):
#         super(GoogleContrastiveLoss, self).__init__()
#         self.

#     def resize_and_project(features, resize_size):
#         resized_features = F.interpolate(features, resize_size, mode='bilinear', align_corners=True)

#         return 

#     def forward():


