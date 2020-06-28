#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

NUM_CLASSES = 10

PROJECT = os.path.join(os.path.dirname(__file__), os.pardir)

ROOT = os.path.join(PROJECT, os.pardir,"data")

# trainig img dir
TRAIN_IMG_DIR = os.path.join(ROOT, "train")
TRAIN_LABEL_DIR = os.path.join(ROOT, "train.csv")
TEST_IMG_DIR = os.path.join(ROOT, "test")

# enhanced 4tissue fv
FV_DIR = os.path.join(PROJECT, "fv")

if __name__ == "__main__":
    pass
