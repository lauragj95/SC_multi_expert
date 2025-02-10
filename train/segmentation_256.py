# -*- coding: utf-8 -*-
"""
Created on Mon May  6 11:12:36 2024

@author: Laura
"""

import os
import tqdm
import time
import wandb
import torch
import torch.nn as nn
import hydra
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from omegaconf import DictConfig
from tensorboardX import SummaryWriter
from source.models import HIPT
from source import dataset_gleason
from source.utils import (
    initialize_wandb,
    train_on_regions,
    tune_on_regions,
    compute_time,
    update_log_dict,
    EarlyStopping,
    OptimizerFactory,
    SchedulerFactory,
    make_weights_for_balanced_classes
)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=2,
                    help='batch size') 
parser.add_argument('--valid_batch_size', type=int, default=2,
                    help='valid batch size')
parser.add_argument('--num_epochs', type=int, default=70,
                    help='Max number of epochs')
parser.add_argument('--n_fold', type=int, default=0,
                    help='n fold')
parser.add_argument('--expert', type=int, default=1,
                    help='Expert')
parser.add_argument('--nclasses', type=int, default=5,
                    help='Expert')
parser.add_argument('--json_file', type=str, default='kfolds.json',
                    help='json file')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='Initial learning rate')
parser.add_argument('--data_dir', type=str,default = "D:/datasets/Gleason19",
                    help='Data path')
parser.add_argument('--experiment_name', type=str, default='segmentation', 
                    help='run name')
parser.add_argument('--save_dir', type=str, default='runs', 
                    help='Directory to save the models to')
parser.add_argument('--early_stopping', type=bool, default=True, 
                    help='early stopping')
parser.add_argument('--checkpoint_256', type=str, default='./checkpoints/vit_256_small_dino_fold_4.pt', 
                    help='checkpoint 256')
parser.add_argument('--checkpoint_4k', type=str, default='./checkpoints/vit_4096_xs_dino_fold_4.pt', 
                    help='checkpoint 4k')
args = parser.parse_args()

def main(args):
    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
    # set up wandb
    print(torch.cuda.get_device_name())

    writer = SummaryWriter(f'{args.save_dir}/{args.experiment_name}')
    output_dir = Path(f'{args.save_dir}/{args.experiment_name}')
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(f'{output_dir}/checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    result_dir = Path(f'{output_dir}/results')
    result_dir.mkdir(parents=True, exist_ok=True)


    criterion = nn.CrossEntropyLoss()

    model = HIPT(num_classes = args.nclasses, pretrain_vit_patch = args.checkpoint_256, pretrain_vit_region = args.checkpoint_4k)
    model.relocate(gpu_id = 0)
    print(model)
    
    
    ###### PREPARE DATA #######
    print(f"Loading data")
    dataset_loading_start_time = time.time()
    train_data = dataset_gleason.get_Gleason1(args.data_dir, args.n_fold, args.json_file, args.expert, train = True)
    valid_data = dataset_gleason.get_Gleason1(args.data_dir, args.n_fold, args.json_file, args.expert, train = False)
    
    train_dataset = dataset_gleason.GleasonDataset(train_data, True)
    valid_dataset = dataset_gleason.GleasonDataset(valid_data, False)
    
    weights = make_weights_for_balanced_classes(train_data[1],args.nclasses)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        num_workers=0, sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights)),drop_last=True)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.valid_batch_size,
        num_workers=0, shuffle=False,drop_last=True)



    

    ###### HYPERPARAMETERS
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(
        lr=args.lr,amsgrad=True, params=model.parameters(), eps=1e-7, weight_decay=1e-7)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    criterion = nn.CrossEntropyLoss()



    stop = False
    start_time = time.time()

    with tqdm.tqdm(
        range(args.num_epochs),
        desc=(f"HIPT Training"),
        unit=" epoch",
        ncols=100,
        leave=True,
    ) as t:
        for epoch in t:
            epoch_start_time = time.time()

            # set dataset seed
            train_dataset.seed = epoch
            valid_dataset.seed = epoch

            train_on_regions(
                epoch + 1,
                model,
                train_loader,
                optimizer,
                criterion,
                batch_size=args.batch_size,
                weighted_sampling=False,
                gradient_accumulation=32,
                num_workers=0,
                writer = writer
            )

            
            tune_on_regions(
                epoch + 1,
                model,
                valid_loader,
                criterion,
                batch_size=args.valid_batch_size,
                num_workers=0,
                writer=writer
            )


            lr = args.lr
            if scheduler:
                lr = scheduler.get_last_lr()
                scheduler.step()



            epoch_end_time = time.time()
            epoch_mins, epoch_secs = compute_time(epoch_start_time, epoch_end_time)
            tqdm.tqdm.write(
                f"End of epoch {epoch+1} / {args.num_epochs} \t Time Taken:  {epoch_mins}m {epoch_secs}s"
            )
            
            save_path = Path(output_dir, f"epoch_{epoch:03}.pt")
            torch.save(model, save_path)




if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main(args)
