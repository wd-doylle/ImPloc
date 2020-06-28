#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import math
from util import constant as c



def get_gene_list(fv):
    '''not consider train/val/test'''
    return [x.rsplit('.')[0] for x in os.listdir(c.TRAIN_IMG_DIR)]


def load_gene_label():
    return _load_label_from_file(c.TRAIN_LABEL_DIR)

def _load_label_from_file(label_file):
    d = {}
    with open(label_file, 'r') as f:
        for line in f.readlines():
            gene, label = line.strip("\n").split(",")
            labels = [int(x) for x in label.split(";") if x]
            if labels:
                d[gene] = labels
    return d


def get_test_gene_list():
    return [x.rsplit('.')[0] for x in os.listdir(c.TEST_IMG_DIR)]


def shuffle(items, batch=128):
    '''shuffle & split into batch'''
    length = len(items)
    index = list(range(len(items)))
    random.shuffle(index)
    a = [items[i] for i in index]
    return [a[i:i+batch] for i in range(0, length, batch)]







if __name__ == "__main__":
    # label_stat(0)
    # label_stat(1)
    # label_stat(2)
    # print(get_label_freq(2))
    # test_global_balance()
    count_img()
