#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torchvision
import os
import sys
from PIL import Image
import numpy as np
import time
# import gpustat
import threading
import queue

sys.path.append(os.path.dirname(__file__))
import finetune
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from util import constant as c


# for enhanced level data
DATA_DIR = c.TRAIN_IMG_DIR
FV_DIR = c.FV_DIR
NUM_THREADS = 1
LAYER = {
    128:-4,
    256:-3,
    512:-2
}
SAVE_DIR = None

if not os.path.exists(FV_DIR):
    os.mkdir(FV_DIR)

# def get_gpu_usage(device=1):
#     gpu_stats = gpustat.new_query()
#     item = gpu_stats.jsonify()["gpus"][device]
#     return item['memory.used'] / item['memory.total']


def extract_image_fv(q, model, save_dir):

    def _extract_image(image):

        img = Image.open(image)
        # img = cv2.resize(img, (3000, 3000), interpolation=cv2.INTER_CUBIC)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        # return img

        inputs = torch.from_numpy(img).type(torch.cuda.FloatTensor)
        pd = model(inputs)
        return pd

    while not q.empty():

        # while get_gpu_usage() > 0.9:
        #     print("---gpu full---", get_gpu_usage())
        #     time.sleep(1)
        #     torch.cuda.empty_cache()

        gene = q.get()
        print("---extract -----", gene)
        gene_dir = os.path.join(DATA_DIR, gene)
        outpath = os.path.join(save_dir,"%s.pkl" % gene)
        if os.path.exists(outpath):
            print("------already extracted---------", gene)
            continue

        pds = [_extract_image(os.path.join(gene_dir, p))
               for p in os.listdir(gene_dir)
               if os.path.splitext(p)[-1] == ".jpg"]
        if pds:
            value = torch.cat(pds,dim=0)
            # print(value.shape)
            print("----save-----", outpath)
            torch.save(value, outpath)


def extract(fv_dim=128):
    q = queue.Queue()
    for gene in os.listdir(DATA_DIR):
        q.put(gene)
    model = finetune.finetune_model()
    model = finetune.FineTuneModel(model,layer=LAYER[fv_dim])
    for param in model.parameters():
        param.requires_grad = False
    model.share_memory()
    model.cuda()
    # print(model)

    save_dir = os.path.join(FV_DIR,"res18-%d"%fv_dim)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    jobs = []
    for i in range(NUM_THREADS):
        p = threading.Thread(target=extract_image_fv, args=(q, model,save_dir))
        jobs.append(p)
        p.daemon = True
        p.start()

    for j in jobs:
        j.join()


if __name__ == "__main__":
    extract()
