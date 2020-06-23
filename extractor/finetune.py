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


def finetune_model():
    root_dir = os.path.dirname(__file__)
    model_dir = os.path.join(root_dir,'finetune')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_path = os.path.join(model_dir,'resnet-18.pth')

    model = torchvision.models.resnet18()
    model.fc = torch.nn.Linear(512,c.NUM_CLASSES)
    model = torch.nn.Sequential(model,torch.nn.Sigmoid())
    model.to('cuda')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        return list(model.children())[0]
    print("Finetuning ResNet-18...")
    criterion = torch.nn.BCELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.001, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5,
            patience=50, min_lr=1e-5)
    writer = SummaryWriter(model_dir)
    train_loader = list(get_train_loader())
    val_loader = list(get_val_loader())
    epochs = 1000
    for e in range(1,epochs+1):
        model.train()
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


def get_train_loader(batch_size=4):
    items = []
    gene_label = datautil.load_gene_label()
    for gene in os.listdir(c.TRAIN_IMG_DIR):
        gene_dir = os.path.join(c.TRAIN_IMG_DIR, gene)
        labels = gene_label[gene]
        for p in os.listdir(gene_dir)[:2]:
            img = Image.open(os.path.join(gene_dir,p))
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0)
            input = torch.from_numpy(img).type(torch.FloatTensor)
            label = torch.zeros((1,c.NUM_CLASSES))
            label[0,gene_label[gene]] = 1
            items.append([input,label])
    
    batched = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]
    for batch in batched:
        (inputs, labels) = zip(*batch)
        yield(torch.cat(inputs),torch.cat(labels))

def get_val_loader(batch_size=4):
    items = []
    gene_label = datautil.load_gene_label()
    for gene in os.listdir(c.TRAIN_IMG_DIR):
        gene_dir = os.path.join(c.TRAIN_IMG_DIR, gene)
        labels = gene_label[gene]
        for p in os.listdir(gene_dir)[2:3]:
            img = Image.open(os.path.join(gene_dir,p))
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0)
            input = torch.from_numpy(img).type(torch.FloatTensor)
            label = torch.zeros((1,c.NUM_CLASSES))
            label[0,gene_label[gene]] = 1
            items.append([input,label])
    
    batched = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]
    for batch in batched:
        (inputs, labels) = zip(*batch)
        yield(torch.cat(inputs),torch.cat(labels))