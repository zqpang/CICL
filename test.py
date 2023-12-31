from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random

import numpy as np
import numpy
import sys


import time
from datetime import timedelta

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader  

from cicl import datasets
from cicl import models
from cicl.models.embeddingmodel import Fusion_model

from cicl.evaluators import Evaluator, extract_features
from cicl.utils.data import transforms as T
from cicl.utils.data.preprocessor import Preprocessor
from cicl.utils.logging import Logger
from cicl.utils.serialization import load_checkpoint


import os


def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    train = True
    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))
        train = False

    test_loader = DataLoader(
        Preprocessor(testset, train, root=None, transform1=test_transformer,transform2 = test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
    return test_loader


def create_model(args):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout, num_classes=0)
    model.cuda()
    model = nn.DataParallel(model)
    return model

def evaluate_mean(evaluator1, dataset, test_loaders):
    maxap = 0
    maxcmc = 0
    mAP_sum = 0
    cmc_sum = 0
    cmc_sum_10 = 0

    for i in range(len(dataset)):
        cmc_scores, mAP = evaluator1.evaluate(test_loaders[i], dataset[i].query, dataset[i].gallery, cmc_flag=False)
        maxap = max(mAP, maxap)
        maxcmc = max(cmc_scores[0], maxcmc)
        mAP_sum += mAP
        cmc_sum += cmc_scores[0]
        cmc_sum_10 += cmc_scores[9]

    mAP = (mAP_sum) / len(test_loaders)
    cmc_now = (cmc_sum) / len(test_loaders)
    cmc_now_10 = cmc_sum_10 / (len(test_loaders))

    return mAP, cmc_now, cmc_now_10


def main():    
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    main_worker(args)
    




def main_worker(args):
    
    start_time = time.monotonic()
    cudnn.benchmark = True
    
    sys.stdout = Logger(osp.join(args.logs_dir, args.dataset, 'test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    #args.data_dir = '/root/pxu1/datasets/{}_all'.format(args.dataset)
    dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers)
    
    if args.dataset == 'prcc':
        datasets_prcc = []
        test_loaders_prcc = []

        for _ in range(10):
            dataset_cur = get_data(args.dataset, args.data_dir)
            test_loader_cur = get_test_loader(dataset_cur, args.height, args.width, args.batch_size, args.workers)
            datasets_prcc.append(dataset_cur)
            test_loaders_prcc.append(test_loader_cur)
    
    
   
    
    model_rgb = create_model(args)
    model_fusion = Fusion_model(model_rgb.module.num_features)
    model_fusion.cuda()
    model_fusion = torch.nn.DataParallel(model_fusion)
    
    evaluator1 = Evaluator(model_rgb)
    
    

        
    params = []
    print('prepare parameter')
    
    models = [model_rgb, model_fusion]
    for model in models:
        for key, value in model.named_parameters():
            if value.requires_grad:
                params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
    
        
        
    print ('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, args.dataset, 'model_best.pth.tar'))
    model_rgb.load_state_dict(checkpoint['state_dict'])
    
    if args.dataset == 'prcc':
        mAP, cmc_now, cmc_now_10 = evaluate_mean(evaluator1, datasets_prcc, test_loaders_prcc)
    else:
        cmc_socore1,mAP1 = evaluator1.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)
        mAP, cmc_now, cmc_now_10  = mAP1, cmc_socore1[0], cmc_socore1[9]
        
    
    print('=================RGB===================')
    print('the RGB model performance')
    print('model mAP: {:5.1%}'.format(mAP))
    print('model cmc: {:5.1%}'.format(cmc_now))
    print('model cmc_10: {:5.1%}'.format(cmc_now_10))
    print('===============================================')
    
    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CICL")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='VC_Clothes',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.60,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--output_weight', type=float, default=1.0,
                        help="loss outputs for weight ")
    parser.add_argument('--ratio_cluster', type=float, default=1.0,
                        help="cluster hypter ratio ")
    parser.add_argument('--cr', action="store_true", default=False,
                        help="use cluster refinement in CACL")
    
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the hybrid memory")
    parser.add_argument('--loss-size', type=int, default=2)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--step-size', type=int, default=20)
    parser.add_argument('--sic_weight', type=float, default=1,
                        help="loss outputs for sic ")
    # training configs
    parser.add_argument('--seed', type=int, default=1)#
    parser.add_argument('--print-freq', type=int, default=200)
    parser.add_argument('--eval-step', type=int, default=5)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    data_dir = "/home/zhiqi/dataset"
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=data_dir)
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument("--cuda", type=str, default="2,3", help="cuda")
    main()
