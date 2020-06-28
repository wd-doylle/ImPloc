#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import os
from model import mtrain
from model import mpredict
from model import img_rnn
from util import torch_util

# matlab fv use batchsize=64
# batchsize = 256
batchsize = 16


def transformer_bce(fv, fold=1,num_heads=2,hid_dim=32,num_layers=4):
    model_name = "transformer_%s_bce_fold%d_numheads%d_hiddim%d-numlayers%d" % (fv, fold, num_heads, hid_dim, num_layers)
    criterion = torch.nn.BCELoss(reduction='none')
    return mtrain.train(fv, model_name, criterion, batchsize=batchsize,fold=fold,num_heads=num_heads,hid_dim=hid_dim,num_layers=num_layers)

def GAT_bce(fv, fold=1,num_heads=2,hid_dim=32):
    model_name = "GAT_%s_bce_fold%d_numheads%d_hiddim%d" % (fv, fold, num_heads, hid_dim)
    criterion = torch.nn.BCELoss(reduction='none')
    return mtrain.train(fv, model_name, criterion, batchsize=batchsize,fold=fold,num_heads=num_heads,hid_dim=hid_dim)

def transformer_predict(fv,model_path,num_heads=2,hid_dim=32):
    model_name = os.path.dirname(model_path)
    mpredict.predict(fv,model_name,model_path,batchsize=batchsize,num_heads=num_heads,hid_dim=hid_dim)

if __name__ == "__main__":
    # transformer_bce("res18-64")
    # transformer_bce("res18-128")
    # transformer_bce("res18-256")
    # transformer_bce("res18-512")
    # transformer_bce("matlab")
    # transformer_bce_gbalance("matlab")
    # transformer_fec1("res18-64")
    # transformer_fec1("res18-128")
    # transformer_fec1("res18-256")
    # transformer_fec1("res18-512")
    # transformer_fec5("matlab")
    # fix_break_run()
    # imgrnn_bce_kfold()
    pass