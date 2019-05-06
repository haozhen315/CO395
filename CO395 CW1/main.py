#!/usr/bin/env python

import numpy as np
from plotTree import *
from decisionTree import DecisionTree
import matplotlib.pyplot as plt
from evaluation import *

__author__ = "Jiajun Chen, Peizhen Wu, Kai Zhang, Zhen Hao"
__date__ = "02/10/2019"



def evaluate(dataset):
    '''
    Main function to call to run the processes automatically given a dataset
    '''
    n_folds = 10
    dtree = DecisionTree(dataset[:,:7],dataset[:,-1])
    fold_idx = cross_validation_split(dataset, n_folds)
    print('Cross validation on pre-pruned tree:\n')
    cm = cross_valid(dataset,fold_idx)
    evaluation_metrics(cm)
    print('\nCross validation on post-pruned tree:\n')
    cm_prune = cross_valid_prune(dataset,fold_idx)
    evaluation_metrics(cm_prune)


if __name__ == '__main__':
    np.random.seed(42)
    data_clean = np.loadtxt('wifi_db/clean_dataset.txt')
    data_noisy = np.loadtxt('wifi_db/noisy_dataset.txt')

    evaluate(data_clean)
    evaluate(data_noisy)

    # Uncomment for tree visualization
    # dtree = DecisionTree(data_clean[:,:7],data_clean[:,-1])
    # createPlot(dtree)
