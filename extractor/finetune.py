#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torchvision
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import random
from PIL import Image
import os
from util import constant as c
from util import datautil
import numpy as np



class FineTuneModel(torch.nn.Module):

    def __init__(self, original_model,layer):
        super(FineTuneModel, self).__init__()
        self.features = torch.nn.Sequential(
                        *list(original_model.children())[:layer])
        # print(list(original_model.children())[-4])
        for p in self.features.parameters():
            p.require_grad = False

    def forward(self, x):
        f = self.features(x)
        f = torch.nn.AdaptiveAvgPool2d(1)(f)
        return f.view(f.size(0), -1)


def finetune_model(arch):
    root_dir = os.path.dirname(__file__)
    model_dir = os.path.join(root_dir,'finetune')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if arch == 'resnet18':
        model_path = os.path.join(model_dir,'resnet-18.pth')
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(512,c.NUM_CLASSES)
        train_loader = get_data_loader(stage='train')
        val_loader = get_data_loader(stage='val')
    elif arch == 'resnext':
        model_path = os.path.join(model_dir,'resnext.pth')
        model = torch.hub.load('pytorch/vision:v0.6.0', 'resnext50_32x4d', pretrained=True)
        model.fc = torch.nn.Linear(2048,c.NUM_CLASSES)
        train_loader = get_data_loader(stage='train',batch_size=1)
        val_loader = get_data_loader(stage='val',batch_size=1)
    model = torch.nn.Sequential(model,torch.nn.Sigmoid())
    model.to('cuda')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        return list(model.children())[0]
    print("Finetuning %s..."%arch)
    criterion = torch.nn.BCELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.001, weight_decay=0.001)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5,
            patience=50, min_lr=1e-5)
    writer = SummaryWriter(model_dir)
    epochs = 100
    for e in range(1,epochs+1):
        model.eval()
        print("------epoch--------", e)
        train_losses = []
        for imgs,labels in train_loader:
            model.zero_grad()
            inputs,labels = imgs.cuda(),labels.cuda()
            pd = model(inputs)
            loss = criterion(pd, labels)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
        writer.add_scalar("train loss", torch.mean(torch.tensor(train_losses)), e)

        model.eval()
        val_losses = []
        for imgs,labels in val_loader:
            inputs,labels = imgs.cuda(),labels.cuda()
            pd = model(inputs)
            loss = criterion(pd, labels)
            val_losses.append(loss.item())
        val_loss = torch.mean(torch.tensor(val_losses))
        scheduler.step(val_loss)
        writer.add_scalar("val loss", val_loss, e)
    
    torch.save(model.state_dict(),model_path)

    return list(model.children())[0]


def get_data_loader(batch_size=4,stage='train',val_portion=0.1):
    items = []
    gene_label = datautil.load_gene_label()
    for gene in os.listdir(c.TRAIN_IMG_DIR):
        gene_dir = os.path.join(c.TRAIN_IMG_DIR, gene)
        labels = gene_label[gene]
        dirs = os.listdir(gene_dir)
        dirs = dirs[:int(len(dirs)*val_portion)] if stage == 'train' else dirs[int(len(dirs)*val_portion):]
        for p in dirs:
            img = Image.open(os.path.join(gene_dir,p))
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0)
            input = torch.from_numpy(img).type(torch.FloatTensor)
            label = torch.zeros((1,c.NUM_CLASSES))
            label[0,gene_label[gene]] = 1
            items.append([input,label])
            if len(items)>=batch_size:
                (inputs, labels) = zip(*items)
                items = []
                yield(torch.cat(inputs),torch.cat(labels))
