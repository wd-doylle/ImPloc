#!/usr/bin/env python
# -*- coding: utf-8 -*-

# multi instance train model
import tensorboardX
import time
import os
import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir,'util'))
from util import torch_util
from util import constant as c
sys.path.append(os.path.dirname(__file__))
from transformer import Transformer
import fvloader


def predict(fv, model_name, model_pth, balance=False,
          batchsize=16, num_heads=2, hid_dim=32):

    dloader = fvloader

    test_data = dloader.load_test_data(fv=fv)
    model_dir = os.path.join("./modeldir-revision/%s-%d" % (model_name,time.time()))


    model = Transformer(fv,num_classes=c.NUM_CLASSES,NUM_HEADS=num_heads, hid_dim=hid_dim).cuda()
    model = nn.DataParallel(model)
    if os.path.exists(model_pth):
        print("------load model--------")
        model.load_state_dict(torch.load(model_pth))
    else:
        raise Exception("Model path does not exist!")

    model.eval()
    with open('test-pred.csv','w') as pf:
        with open('test.csv','w') as of:
            for item in fvloader.batch_fv(test_data, batchsize=batchsize):
                genes, nimgs, _, timesteps = item
                inputs = nimgs.cuda()
                pd = model(inputs)
                bin_pd = torch_util.threshold_tensor_batch(pd).cpu()

                for i,gene in enumerate(genes):
                    pf.write("%s,"%gene)
                    of.write("%s,"%gene)
                    for j,p in enumerate(pd[i]):
                        pf.write("%.4f;"%p)
                        if bin_pd[i,j]:
                            of.write("%d;"%j)
                    pf.write("\n")
                    of.write("\n")

if __name__ == "__main__":
    # train("res18-128", 0)
    # train("matlab", 0)
    pass
