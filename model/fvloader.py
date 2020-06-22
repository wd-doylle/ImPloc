#!/usr/bin/env python
# -*- coding: utf-8

import os
import random
import numpy as np
from util import datautil
from util import constant as c
import torch

NUM_CLASSES = 10


def kfold_split(fold=1, fv='res18-128'):
    candidate_genes = datautil.get_gene_list(fv)
    # candidate_genes = candidate_genes[:int(len(candidate_genes) * 0.9)]
    cl = len(candidate_genes)
    vl = cl // 10
    val_start = vl * (fold - 1)
    val_end = vl * fold
    val_genes = set(candidate_genes[val_start:val_end])
    train_genes = set(candidate_genes) - set(val_genes)
    val_genes = list(val_genes)
    train_genes = list(train_genes)

    return train_genes, val_genes


def load_kfold_train_data(fold=1, fv='res18-128'):
    train_genes, val_genes = kfold_split(fold,fv=fv)
    return _load_data(train_genes, fv=fv)


def load_kfold_val_data(fold=1, fv='res18-128'):
    train_genes, val_genes = kfold_split(fold)
    return _load_data(val_genes, fv=fv)


def load_kfold_test_data(fold=1, fv='res18-128'):
    return load_test_data(size=0, fv=fv)


def load_test_data(size=1, fv='res18-128'):
    gene_list = datautil.get_test_gene_list(size)
    return _load_data(gene_list, fv=fv)


def _handle_load(gene, d, fv='res18-128'):
    genef = os.path.join(c.FV_DIR,fv, "%s.pkl" % gene)
    nimg = torch.load(genef).cpu()
    gene_label = torch.zeros(NUM_CLASSES)
    for l in d[gene]:
        gene_label[l] = 1
    timestep = nimg.shape[0]
    return (gene, nimg, gene_label, timestep)


def _load_data(gene_list, fv='res18-128'):
    d = datautil.load_gene_label()
    q = [x for x in gene_list if x in d]

    return [_handle_load(x, d, fv) for x in q]


def shuffle(items):
    index = list(range(len(items)))
    random.shuffle(index)
    return [items[i] for i in index]


def shuffle_with_idx(items):
    idx = np.random.permutation(range(len(items)))
    sitems = [items[x] for x in idx]
    return sitems, idx


def batch_fv(items, batchsize=32):
    length = len(items)
    batched = [items[i:i+batchsize] for i in range(0, length, batchsize)]
    for batch in batched:
        (genes, nimgs, labels, timesteps) = zip(*batch)

        labels = [lab.view(1,-1) for lab in labels]

        maxt = np.max(timesteps)
        pad_imgs = []
        for img in nimgs:
            pad = torch.nn.functional.pad(img, (0,0,0,maxt-img.shape[0]),
                         mode='constant', value=0)
            pad_imgs.append(pad.view(1,maxt,-1))
        yield (genes, torch.cat(pad_imgs),
               torch.cat(labels), timesteps)


def test():
    train_data = load_kfold_test_data()

    # for item in train_data:
    #     print(item[0])
    #     print(item[1].shape)
    #     print(item[2])
    #     print(item[3])

    for batch_data in batch_fv(shuffle(train_data), 4):
        print(batch_data)


if __name__ == "__main__":
    test()
