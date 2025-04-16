import torch
from torch import distributed
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from utils.loss import *
from utils.KD_loss import *
from utils.AttentionLoss import SpatialGatingDistllationLoss, MetaWeightingDistillationLoss
from .segmentation_module import make_model
from modules.classifier import IncrementalClassifier, CosineClassifier, SPNetClassifier
from .utils import get_scheduler, MeanReduction, printCUDA

import numpy as np

import json
import pickle as pkl
import os.path as osp

import random
from PIL import Image

CLIP = 10


class Trainer:
    def __init__(self, task, device, logger, opts):
        self.logger = logger
        self.device = device
        self.task = task
        self.opts = opts
        self.novel_classes = self.task.get_n_classes()[-1]
        self.step = task.step

        self.need_model_old = (opts.born_again or opts.mib_kd > 0 or opts.loss_kd > 0 or
                               opts.l2_loss > 0 or opts.l1_loss > 0 or opts.cos_loss > 0)

        self.n_channels = -1  # features size, will be initialized in make model
        self.model = self.make_model()
        self.model = self.model.to(device)

        self.distributed = False
        self.model_old = None

        self.EMA = opts.EMA
        self.feature_model_old = None

        if opts.fix_bn:
            self.model.fix_bn()

        if opts.bn_momentum is not None:
            self.model.bn_set_momentum(opts.bn_momentum)

        self.initialize(opts)  # initialize model parameters (e.g. perform WI)

        self.born_again = opts.born_again
        self.save_memory = opts.memory
        self.dist_warm_start = opts.dist_warm_start
        model_old_as_new = opts.born_again or opts.dist_warm_start
        if self.need_model_old:
            self.model_old = self.make_model(is_old=not model_old_as_new)
            # put the old model into distributed memory and freeze it
            for par in self.model_old.parameters():
                par.requires_grad = False
            self.model_old.to(device)
            self.model_old.eval()
            # if self.EMA:
            #     self.feature_model_old = self.make_model(is_old= not model_old_as_new)
            #     for par in self.feature_model_old.parameters():
            #         par.requires_grad = False
            #     self.feature_model_old.to(device)
            #     self.feature_model_old.eval()

        # xxx Set up optimizer
        self.train_only_novel = opts.train_only_novel
        params = []
        if not opts.freeze:
            params.append({"params": filter(lambda p: p.requires_grad, self.model.body.parameters())})
        else:
            for par in self.model.body.parameters():
                par.requires_grad = False

        if opts.lr_head != 0:
            params.append({"params": filter(lambda p: p.requires_grad, self.model.head.parameters()),
                           'lr': opts.lr * opts.lr_head})
        else:
            for par in self.model.head.parameters():
                par.requires_grad = False

        if opts.method != "SPN":
            if opts.train_only_novel:
                params.append({"params": filter(lambda p: p.requires_grad, self.model.cls.cls[task.step].parameters()),
                              'lr': opts.lr * opts.lr_cls})
            else:
                params.append({"params": filter(lambda p: p.requires_grad, self.model.cls.parameters()),
                               'lr': opts.lr * opts.lr_cls})

        if opts.contrast_loss > 0:
            params.append({"params": filter(lambda p: p.requires_grad, self.model.proj_head.parameters()),
                            'lr': opts.lr * 0.1})
            for par in self.model.proj_head.parameters():
                par.requires_grad = False

        # params.append({"params": filter(lambda p: p.requires_grad, [self.model.class_importance]),
        #                 'lr': opts.lr})
        # params.append({"params": filter(lambda p: p.requires_grad, [self.model.channel_importance]),
        #                 'lr': opts.lr * 1000})
        # params.append({"params": filter(lambda p: p.requires_grad, [self.model.spatial_importance]),
        #                 'lr': opts.lr * 1000})
        # params.append({"params": filter(lambda p: p.requires_grad, [self.model.centers]),
        #                 'lr': opts.lr * 10})

        self.optimizer = torch.optim.SGD(params, lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
        self.scheduler = get_scheduler(opts, self.optimizer)
        self.logger.debug("Optimizer:\n%s" % self.optimizer)

        reduction = 'none'
        if opts.mib_ce:
            self.criterion = UnbiasedCrossEntropy(old_cl=len(self.task.get_old_labels()),
                                                  ignore_index=255, reduction=reduction)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction=reduction)

        self.reduction = MeanReduction()

        if opts.contrast_loss > 0:
            self.contrast_loss = opts.contrast_loss
            self.contrast_criterion = PixelContrastLoss(temperature=0.07, base_temperature=0.07)
        else:
            self.contrast_criterion = None

        # Feature distillation
        if opts.l2_loss > 0 or opts.cos_loss > 0 or opts.l1_loss > 0:
            assert self.model_old is not None, "Error, model old is None but distillation specified"
            if opts.l2_loss > 0:
                self.feat_loss = opts.l2_loss
                self.feat_criterion = nn.MSELoss()
                # self.feat_criterion = RKDLoss()
                # self.feat_criterion = SimilarityKDLoss()
                # self.feat_criterion = AttentionKD()
                # self.feat_criterion = ABLoss()
            elif opts.l1_loss > 0:
                self.feat_loss = opts.l1_loss
                self.feat_criterion = nn.L1Loss()
            elif opts.cos_loss > 0:
                self.feat_loss = opts.cos_loss
                self.feat_criterion = CosineLoss()
                # self.feat_criterion = SpatialGatingDistllationLoss(opts.n_feat, device=device)
                # self.feat_criterion = MetaWeightingDistillationLoss(opts.n_feat, device=device)
        else:
            self.feat_criterion = None

        if opts.embedding_loss > 0:

            self.embedding_loss = opts.embedding_loss
            # self.embedding_criterion = nn.MSELoss()
            # self.embedding_criterion = nn.L1Loss()
            self.embedding_criterion = CosineLoss()
        else:
            self.embedding_criterion = None

        # Output distillation
        if opts.loss_kd > 0 or opts.mib_kd > 0:
            assert self.model_old is not None, "Error, model old is None but distillation specified"
            if opts.loss_kd > 0:
                if opts.ckd:
                    self.kd_criterion = CosineKnowledgeDistillationLoss(reduction='mean')
                else:
                    self.kd_criterion = KnowledgeDistillationLoss(reduction="mean", alpha=opts.kd_alpha, device=device)
                self.kd_loss = opts.loss_kd
            if opts.mib_kd > 0:
                self.kd_loss = opts.mib_kd
                self.kd_criterion = UnbiasedKnowledgeDistillationLoss(reduction="mean")
        else:
            self.kd_criterion = None
            torch.nn.CrossEntropyLoss

        # Body distillation
        if opts.loss_de > 0:
            assert self.model_old is not None, "Error, model old is None but distillation specified"
            self.de_loss = opts.loss_de
            self.de_criterion = CosineLoss()
        else:
            self.de_criterion = None

        if opts.mixup_hidden:
            self.mixup_criterion = MyCrossEntropy
        else:
            self.mixup_criterion = None

        # self.centerLoss = CenterLoss(num_classes=21, feature_dim=256)

    def make_model(self, is_old=False):
        classifier, self.n_channels = self.get_classifier(is_old)
        model = make_model(self.opts, classifier)
        return model

    def distribute(self):
        opts = self.opts
        if self.model is not None:
            # Put the model on GPU
            self.distributed = True
            self.model = DistributedDataParallel(self.model, device_ids=[opts.device_id],
                                                 output_device=opts.device_id, find_unused_parameters=True)

    def get_classifier(self, is_old=False):
        # here distinguish methods!
        opts = self.opts
        if opts.method == "SPN":
            classes = self.task.get_old_labels() if is_old else self.task.get_order()
            cls = SPNetClassifier(opts, classes)
            n_feat = cls.channels
        elif opts.method == 'COS':
            n_feat = self.opts.n_feat
            n_classes = self.task.get_n_classes()[:-1] if is_old else self.task.get_n_classes()
            cls = CosineClassifier(n_classes, channels=n_feat)
        else:
            n_feat = self.opts.n_feat
            n_classes = self.task.get_n_classes()[:-1] if is_old else self.task.get_n_classes()
            cls = IncrementalClassifier(n_classes, channels=n_feat)
        return cls, n_feat

    def initialize(self, opts):
        if opts.init_mib and opts.method == "FT":
            device = self.device
            model = self.model.module if self.distributed else self.model

            classifier = model.cls
            imprinting_w = classifier.cls[0].weight[0]
            bkg_bias = classifier.cls[0].bias[0]

            bias_diff = torch.log(torch.FloatTensor([self.task.get_n_classes()[-1] + 1])).to(device)

            new_bias = (bkg_bias - bias_diff)

            classifier.cls[-1].weight.data.copy_(imprinting_w)
            classifier.cls[-1].bias.data.copy_(new_bias)

            classifier.cls[0].bias[0].data.copy_(new_bias.squeeze(0))

    def warm_up(self, dataset, epochs=1):
        self.warm_up_(dataset, epochs)
        # warm start means make KD after weight imprinting or similar
        if self.dist_warm_start:
            self.model_old.load_state_dict(self.model.state_dict())

    def warm_up_(self, dataset, epochs=1):
        pass

    def cool_down(self, dataset, epochs=1):
        pass

    def train(self, cur_epoch, train_loader, metrics=None, print_int=10, n_iter=1, val_interval=20):
        """Train and return epoch loss"""
        if metrics is not None:
            metrics.reset()
        logger = self.logger
        optim = self.optimizer
        scheduler = self.scheduler
        logger.info("Epoch %d, lr = %f" % (cur_epoch, optim.param_groups[0]['lr']))

        device = self.device
        model = self.model
        criterion = self.criterion

        epoch_loss = 0.0
        reg_loss = 0.0
        contrast_loss = 0.0
        interval_loss = 0.0

        model.train()
        if self.opts.freeze and self.opts.bn_momentum == 0:
            model.module.body.eval()
        if self.opts.lr_head == 0 and self.opts.bn_momentum == 0:
            model.module.head.eval()

        cur_step = 0
        for iteration in range(n_iter):
            train_loader.sampler.set_epoch(cur_epoch*n_iter + iteration)  # setup dataloader sampler
            for images, labels, _ in train_loader:

                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)

                loss_tot = torch.tensor([0.]).to(self.device)
                rloss = torch.tensor([0.]).to(self.device)
                cont_loss = torch.tensor([0.]).to(self.device)

                if self.model_old is not None:
                    with torch.no_grad():
                        # 这个顺序，增量阶段不能使用mixup
                        lam = None
                        if self.mixup_criterion:
                            outputs_old, feat_old, body_old, embedding_old, y_a, y_b, lam = self.model_old(images, target=labels, return_feat=True, return_body=True, return_embedding=True)
                        else:
                            outputs_old, feat_old, body_old, embedding_old = self.model_old(images, return_feat=True, return_body=True, return_embedding=True)

                optim.zero_grad()
                # if self.mixup_criterion:
                #     outputs, feat, body, embedding, y_a, y_b, lam = model(images, target=labels, return_feat=True, return_body=True, return_embedding=True)
                # else:
                    # outputs, feat, body, embedding, channel_weights = model(images, return_feat=True, return_body=True, return_embedding=True, old_features=feat_old)
                if self.mixup_criterion:
                    if self.model_old:
                        outputs, feat, body, embedding = model(images, lam=lam, return_feat=True, return_body=True, return_embedding=True)
                    else:
                        outputs, feat, body, embedding, y_a, y_b, lam = model(images, target=labels, return_feat=True, return_body=True, return_embedding=True)
                else:
                    outputs, feat, body, embedding = model(images, return_feat=True, return_body=True, return_embedding=True)
                    # outputs, feat, body, embedding, weight = model(images, lam=lam, return_feat=True, return_body=True, return_embedding=True, old_features=feat_old)

                # xxx Distillation/Regularization Losses
                if self.model_old is not None:
                        # _, feat_old = self.feature_model_old(images, lam=lam, return_feat=True)
                        # if cur_epoch > val_interval and self.EMA:
                        #     _outputs_old, _feat_old, _, _ = self.feature_model_old(images, lam=lam, return_feat=True, return_body=True, return_embedding=True)
                        #     alpha = 0.5
                        #     feat_old = alpha * _feat_old + (1 - alpha) * feat_old
                        #     beta = 0
                        #     outputs_old = beta * _outputs_old + (1 - beta) * outputs_old
                        #     del _outputs_old, _feat_old
                    if self.kd_criterion is not None:
                        #TODO 目前实验没有增加bg这一类别，同时考虑后续调整权重的可能
                        # old_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                        # new_classes = [16, 17, 18, 19, 20]
                        # masks = torch.isin(labels.cpu(), torch.tensor(new_classes)).float()
                        kd_loss = self.kd_loss * self.kd_criterion(outputs, outputs_old)
                        # print(f"kd_loss - {kd_loss} - {self.kd_loss}")
                        rloss += kd_loss
                        del outputs_old
                    if self.feat_criterion is not None:
                        # old_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                        # new_classes = [16, 17, 18, 19, 20]
                        # masks = torch.isin(labels.cpu(), torch.tensor(new_classes)).float()
                        # masks = F.interpolate(masks.unsqueeze(1), size=feat.shape[-2:], mode='nearest')
                        # feat = masks.to(device) * feat
                        # feat_old = masks.to(device) * feat_old
                        feat_loss = self.feat_loss * self.feat_criterion(feat, feat_old)
                        # print(f"feature cos loss - {feat_loss}")
                        del feat_old
                        if feat_loss <= CLIP:
                            rloss += feat_loss
                        else:
                            print(f"Warning, feature loss is {feat_loss}! Term ignored")
                    if self.de_criterion is not None:
                        de_loss = self.de_loss * self.de_criterion(body, body_old)
                        # print(f"de_loss - {de_loss}")
                        del body_old
                        rloss += de_loss
                    if self.embedding_criterion is not None:
                        embedding_loss = self.embedding_loss * self.embedding_criterion(embedding, embedding_old)
                        del embedding_old
                        rloss += embedding_loss

                if self.contrast_criterion is not None:
                    tmp = self.contrast_loss * self.contrast_criterion(labels, outputs, embedding)
                    # print(f"contrast_loss - {tmp}")
                    rloss += tmp

                if self.mixup_criterion is not None:
                    loss = self.mixup_criterion(outputs, y_a, y_b, lam, num_classes=21, ignore_index=255, reduction='mean')
                else:
                    loss = self.reduction(F.cross_entropy(outputs, labels, ignore_index=255, reduction='none'), labels)

                # Since we noted sometimes that rloss spikes, we added a CLIP value to remove it if higher
                if rloss <= CLIP:
                    loss_tot = loss_tot + loss + rloss
                else:
                    print(f"Warning, rloss is {rloss}! Term ignored")
                    loss_tot += loss

                # center_loss = self.centerLoss(feat, labels, model.module.cls.get_centers())
                # print(f"center_loss - {center_loss}")
                # loss_tot += center_loss

                if cont_loss <= CLIP:
                    loss_tot += cont_loss
                else:
                    print(f"Warning, cont_loss is {cont_loss}! Term ignored")

                # loss = kd_loss + feat_loss
                # loss.backward()

                loss_tot.backward()
                optim.step()
                scheduler.step()

                epoch_loss += loss.item()
                reg_loss += rloss.item()
                interval_loss += loss_tot.item()
                contrast_loss += cont_loss.item()

                _, prediction = outputs.max(dim=1)  # B, H, W
                labels = labels.cpu().numpy()
                prediction = prediction.cpu().numpy()
                if metrics is not None:
                    metrics.update(labels, prediction)

                cur_step += 1
                if cur_step % 10 == 1 and self.device is torch.device('cuda:0'):
                    if self.de_criterion:
                #     print(f"embedding_loss - {embedding_loss}")
                        print(f"body_loss - {de_loss}")
                    if self.contrast_criterion:    
                        print(f"contrast_loss - {tmp}")
                    if self.feat_criterion:
                        print(f"feature loss - {feat_loss}")

                if cur_step % print_int == 0:
                    # print(f"loss_tot: {loss_tot}, class loss - {loss}, reg loss - {rloss}, contrast loss - {cont_loss}")
                    interval_loss = interval_loss / print_int
                    logger.info(f"Epoch {cur_epoch}, Batch {cur_step}/{n_iter}*{len(train_loader)},"
                                f" Loss={interval_loss} RL={reg_loss/print_int} CL={contrast_loss/print_int}]")
                    logger.debug(f"Loss made of: CE {loss}")
                    # visualization
                    if logger is not None:
                        x = cur_epoch * len(train_loader) * n_iter + cur_step
                        logger.add_scalar('Loss', interval_loss, x)
                    interval_loss = 0.0
                    reg_loss = 0.0
                    contrast_loss = 0.0

                del outputs, feat, body, embedding

        # collect statistics from multiple processes
        epoch_loss = torch.tensor(epoch_loss).to(self.device)
        reg_loss = torch.tensor(reg_loss).to(self.device)
        contrast_loss = torch.tensor(contrast_loss).to(self.device)

        torch.distributed.reduce(epoch_loss, dst=0)
        torch.distributed.reduce(reg_loss, dst=0)
        torch.distributed.reduce(contrast_loss, dst=0)

        if distributed.get_rank() == 0:
            epoch_loss = epoch_loss / distributed.get_world_size() / (len(train_loader) * n_iter)
            reg_loss = reg_loss / distributed.get_world_size() / (len(train_loader) * n_iter)
            contrast_loss = contrast_loss / distributed.get_world_size() / (len(train_loader) * n_iter)

        # collect statistics from multiple processes
        if metrics is not None:
            metrics.synch(device)

        logger.info(f"Epoch {cur_epoch}, Class Loss={epoch_loss}, Reg Loss={reg_loss}, Contrast Loss={contrast_loss}")

        return epoch_loss, reg_loss, contrast_loss

    def validate(self, loader, metrics, ret_samples_ids=None, novel=False):
        """Do validation and return specified samples"""
        metrics.reset()
        model = self.model
        device = self.device
        criterion = self.criterion
        logger = self.logger

        class_loss = 0.0

        ret_samples = []
        with torch.no_grad():
            model.eval()
            for i, (images, labels, _) in enumerate(loader):

                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)

                outputs = model(images)  # B, C, H, W
                if novel:
                    outputs[:, 1:-self.novel_classes] = -float("Inf")

                loss = criterion(outputs, labels).mean()

                class_loss += loss.item()

                _, prediction = outputs.max(dim=1)  # B, H, W
                labels = labels.cpu().numpy()
                prediction = prediction.cpu().numpy()
                metrics.update(labels, prediction)

                if ret_samples_ids is not None and i in ret_samples_ids:  # get samples
                    ret_samples.append((images[0].detach().cpu().numpy(),
                                        labels[0], prediction[0]))

            # collect statistics from multiple processes
            metrics.synch(device)

            class_loss = torch.tensor(class_loss).to(self.device)

            torch.distributed.reduce(class_loss, dst=0)

            if distributed.get_rank() == 0:
                class_loss = class_loss / distributed.get_world_size() / len(loader)

            if logger is not None:
                logger.info(f"Validation, Class Loss={class_loss}")

        return class_loss, ret_samples
    

    def memory(self, loader, root='data', dataName='voc', topK=1, num_classes=20):
        idxes = {}
        class_to_images = pkl.load(open(f'{root}/{dataName}/split/inverse_dict_train.pkl', 'rb'))
        # images = np.load(f'{root}/{dataName}/split/train_ids.npy')

        novel_images = set()
        for cl, img_set in class_to_images.items():
            if cl not in loader.dataset.labels and (cl != 0):
                novel_images.update(img_set)
        
        print(novel_images)

        for cl in loader.dataset.labels:
            idxes[cl] = random.sample(set(class_to_images[cl].tolist()).difference(novel_images), topK)
        
        return idxes        
    

    def memory(self, loader, root='data', dataName='voc', topK=1, num_classes=20):
        model = self.model
        device = self.device
        model.eval()

        similarity_matrix = []
        img_list = []

        with torch.no_grad():
            for t, (images, labels, names) in enumerate(loader):
                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)
                img_list.extend(list(names))

                outputs, feat = model(images, return_feat=True)
                outputs = F.softmax(outputs, dim=1)

                # outputs = outputs.cpu().numpy()
                _, predictions = outputs.max(dim=1)
                #TODO 用prediction还是label去计算prototype

                for i in range(images.shape[0]):
                    clses = labels[i].unique().cpu().numpy()
                    clses = clses[(clses != 0) & (clses != 255)]
                    matrix = np.zeros((num_classes))
                    for cls in clses:
                        feat = F.interpolate(feat, size=labels.shape[-2:], mode='bilinear', align_corners=False)
                        mask = labels[i] == cls
                        gap_logit = feat[i, :, mask].mean(dim=1)
                        # print(f"logit - {gap_logit}, cls - {model.module.cls.cls[0].weight.data[cls].squeeze().shape}")
                        # gap_logit = outputs[i, cls][predictions[i] == cls].mean()
                        matrix[cls] = F.cosine_similarity(gap_logit.unsqueeze(0), model.module.cls.cls[0].weight.data[cls].squeeze().unsqueeze(0), dim=1)
                    similarity_matrix.append(matrix)

        similarity_matrix = np.stack(similarity_matrix, axis=0)
        cls_to_names = {}
        
        for cls in range(1, num_classes):
            # indices = np.argsort(similarity_matrix[:, cls])[-topK:][::-1].tolist()
            indices = np.argsort(similarity_matrix[:, cls])[:topK][::-1].tolist()
            cls_to_names[cls] = []
            # print(f"class {cls} - {[similarity_matrix[idx, cls] for idx in indices]}")
            for idx in indices:
                name = img_list[idx].split('/')[-1].split('.')[0]
                lbl = np.array(Image.open(f'data/voc/dataset/annotations/{name}.png'))
                # print(f"class {cls} - {np.unique(lbl)}")
                cls_to_names[cls].append(f"images/{name}.jpg")
            # idxes.update(set(indices))

        # print(cls_to_names)

        images = np.load(f'{root}/{dataName}/split/train_list.npy')
        cls_to_indexs = {}
        for cl, names in cls_to_names.items():
            cls_to_indexs[cl] = []
            for name in names:
                cls_to_indexs[cl].append(int(np.where(images[:,0] == name)[0][0]))

        print(cls_to_indexs)
        
        # images = np.load(f'data/voc/split/train_ids.npy')
        # images = [(f'data/voc/dataset/images/{i}.jpg', f'data/voc/dataset/annotations/{i}.jpg') for i, img in enumerate(images) if i in idxes]
        # print(images)

        return cls_to_indexs


    def load_state_dict(self, checkpoint, strict=True):
        state = {}
        if self.need_model_old or not self.distributed:
            for k, v in checkpoint["model"].items():
                state[k[7:]] = v

        model_state = state if not self.distributed else checkpoint['model']

        if self.born_again and strict:
            self.model_old.load_state_dict(state)
            self.model.load_state_dict(model_state)
            # if self.EMA:
            #     self.feature_model_old.load_state_dict(state)
        else:
            if self.need_model_old and not strict:
                self.model_old.load_state_dict(state, strict=not self.dist_warm_start)  # we are loading the old model
                # if self.EMA:
                #     self.feature_model_old.load_state_dict(state, strict=not self.dist_warm_start)

            if 'module.cls.class_emb' in state and not strict:  # if distributed
                # remove from checkpoint since SPNClassifier is not incremental
                del state['module.cls.class_emb']

            if 'cls.class_emb' in state and not strict:  # if not distributed
                # remove from checkpoint since SPNClassifier is not incremental
                del state['cls.class_emb']

            self.model.load_state_dict(model_state, strict=strict)

            if not self.born_again and strict and not self.save_memory:  # if strict, we are in ckpt (not step) so load also optim and scheduler
                print(f"optimizer - {checkpoint['optimizer']['param_groups']}")
                print(f"self optimizer - {self.optimizer['param_groups']}")
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self.scheduler.load_state_dict(checkpoint["scheduler"])

    def state_dict(self):
        state = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                 "scheduler": self.scheduler.state_dict()}
        return state