# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 10:17:03 2024

@author: Laura
"""
import os
import sys
import tqdm
import time
import json
import hydra
import shutil
import random
import datetime
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import pandas as pd
import argparse
from source import dataset_gleason
from pathlib import Path
from omegaconf import DictConfig
from torchvision import datasets
from tensorboardX import SummaryWriter
import source.vision_transformer as vits
from source.utils import initialize_wandb, compute_time, update_log_dict
from source.components import DINOLoss
from pretrain.utils import (
    PatchDataAugmentationDINO,
    MultiCropWrapper,
    EarlyStoppingDINO,
    train_one_epoch,
    tune_one_epoch,
    fix_random_seeds,
    has_batchnorms,
    get_params_groups,
    resume_from_checkpoint,
    cosine_scheduler,
    get_world_size,
    is_main_process,
)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--patch_size', type=int, default=16,
                    help='patch size')
parser.add_argument('--num_epochs', type=int, default=100,
                    help='Max number of epochs')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='Initial learning rate')
parser.add_argument('--data_dir', type=str,default = "D:/datasets/Gleason19/patches/256/imgs",
                    help='Data path')
parser.add_argument('--experiment_name', type=str, default='test', 
                    help='run name')
parser.add_argument('--save_dir', type=str, default='runs', 
                    help='Directory to save the models to')
parser.add_argument('--early_stopping', type=bool, default=True, 
                    help='early stopping')
parser.add_argument('--model_arch', type=str, default='vit_small', 
                    help='model architecture')
parser.add_argument('--checkpoint', type=str, default='./checkpoints/vit_256_small_dino_fold_4.pt', 
                    help='checkpoint')

args = parser.parse_args()

def main(args):
    writer = SummaryWriter(args.save_dir, args.experiment_name)
    output_dir = Path(args.save_dir, args.experiment_name)

    # preparing data
    if is_main_process():
        print(f"Loading data...")

    # ============ preparing tuning data ============

    # transform = PatchDataAugmentationDINO(
    #     (0.14, 1),
    #     (0.05,0.4),
    #     8,
    # )

    # ============ preparing training data ============
    dataset_loading_start_time = time.time()
    train_data = dataset_gleason.get_Gleason(args.data_dir)
    # valid_data = dataset_gleason.get_Gleason(data_dir, n_fold, json_file, train = False)
    
    dataset = dataset_gleason.GleasonDataset(train_data)
    # valid_dataset = dataset_gleason.GleasonDataset(valid_data,False)
    # dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    dataset_loading_end_time = time.time() - dataset_loading_start_time
    total_time_str = str(datetime.timedelta(seconds=int(dataset_loading_end_time)))
    if is_main_process():
        print(f"Pretraining data loaded in {total_time_str} ({len(dataset)} patches)")

    
    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        pin_memory=True, drop_last=True)
    # validloader = torch.utils.data.DataLoader(
    #     valid_dataset, batch_size=args.batch_size, shuffle=False,
    #     pin_memory=True, drop_last=True)

    # building student and teacher networks
    if is_main_process():
        print(f"Building student and teacher networks...")
    student = vits.__dict__['vit_small'](
        patch_size=16,
        drop_path_rate=0.1,
    )
    teacher = vits.__dict__['vit_small'](patch_size=16)
    embed_dim = student.embed_dim

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = MultiCropWrapper(
        student,
        vits.DINOHead(
            embed_dim,
            65536,
            use_bn=False,
            norm_last_layer=False,
        ),
    )
    teacher = MultiCropWrapper(
        teacher,
        vits.DINOHead(
            embed_dim,
            65536,
            use_bn=False,
        ),
    )

    # move networks to gpu

    student, teacher = student.cuda(), teacher.cuda()

    # teacher_without_ddp and teacher are the same thing
    teacher_without_ddp = teacher
    
    

    # teacher and student start with the same weights
    student_sd = student.state_dict()
    nn.modules.utils.consume_prefix_in_state_dict_if_present(student_sd, "module.")
    teacher_without_ddp.load_state_dict(student_sd)

    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False

    # total number of crops = 2 global crops + local_crops_number
    crops_number = 8 + 2
    dino_loss = DINOLoss(
        65536,
        crops_number,
        0.04,
        0.04,
        0,
        args.num_epochs,
    )

    dino_loss = dino_loss.cuda()
    
    

    params_groups = get_params_groups(student)
    optimizer = torch.optim.AdamW(params_groups)

    # for mixed precision training

    fp16_scaler = torch.cuda.amp.GradScaler()

    base_lr = (
    args.lr * (args.batch_size * get_world_size()) / 256.0
    )
    lr_schedule = cosine_scheduler(
        base_lr,
        1e-6,
        args.num_epochs,
        len(trainloader),
        warmup_epochs=10,
    )
    wd_schedule = cosine_scheduler(
        0.04,
        0.4,
        args.num_epochs,
        len(trainloader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = cosine_scheduler(
        0.996, 1, args.num_epochs, len(trainloader)
    )
    if is_main_process():
        print(f"Models built, kicking off training")

    epochs_run = 0

    # leverage torch native fault tolerance
    ckpt_path = Path(args.checkpoint)
    loc = f"cuda:{0}"
    snapshot = torch.load(args.checkpoint, map_location=loc)
    epochs_run = snapshot["epoch"]
    student_state_dict = snapshot["student"]
    student_state_dict = {k.replace("module.", ""): v for k, v in student_state_dict.items()}
    student.load_state_dict(student_state_dict)
    teacher_state_dict = snapshot["teacher"]
    teacher.load_state_dict(teacher_state_dict)
    optimizer.load_state_dict(snapshot["optimizer"])
    dino_loss.load_state_dict(snapshot["dino_loss"])
    if fp16_scaler is not None:
        fp16_scaler.load_state_dict(snapshot["fp16_scaler"])
    if is_main_process():
        print(f"Resuming training from snapshot at epoch {epochs_run}")



    # epochs_run = resume_from_checkpoint(
    #     ckpt_path,
    #     student=student,
    #     teacher=teacher,
    #     optimizer=optimizer,
    #     fp16_scaler=fp16_scaler,
    #     dino_loss=dino_loss,
    # )
    # if is_main_process():
    #     print(f"Resuming training from checkpoint at epoch {epochs_run}")

    early_stopping = EarlyStoppingDINO(
        'loss',
        'min',
        10,
        30,
        checkpoint_dir=output_dir,
        save_every=10,
        verbose=True,
    )

    stop = False
    start_time = time.time()

    with tqdm.tqdm(
        range(args.num_epochs),
        desc=(f"DINO Pretraining"),
        unit=" epoch",
        ncols=100,
        leave=True,
        initial=0,
        total=args.num_epochs,
        file=sys.stdout,
        position=0,
        disable=not is_main_process(),
    ) as t:
        for epoch in t:
            epoch_start_time = time.time()
            

            # training one epoch of DINO
            train_stats = train_one_epoch(
                student,
                teacher,
                teacher_without_ddp,
                dino_loss,
                trainloader,
                optimizer,
                lr_schedule,
                wd_schedule,
                momentum_schedule,
                epoch,
                args.num_epochs,
                fp16_scaler,
                3.0,
                1,
                0,
                writer
            )


            if is_main_process():
                snapshot = {
                    "epoch": epoch,
                    "student": student.state_dict(),
                    "teacher": teacher.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "dino_loss": dino_loss.state_dict(),
                }
                
                if fp16_scaler is not None:
                    snapshot["fp16_scaler"] = fp16_scaler.state_dict()

            # only run tuning on rank 0, otherwise one has to take care of gathering knn metrics from multiple gpus
            tune_results = None


            # save snapshot and log to wandb
            if is_main_process():
                save_path = Path(output_dir, f"epoch_{epoch:03}.pt")
                if (
                    epoch % 10 == 0
                    and not save_path.is_file()
                ):
                    torch.save(snapshot, save_path)


            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
            }
            if is_main_process():
                with open(Path(output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

            epoch_end_time = time.time()
            epoch_mins, epoch_secs = compute_time(epoch_start_time, epoch_end_time)
            if is_main_process():
                tqdm.tqdm.write(
                    f"End of epoch {epoch+1}/{args.num_epochs} \t Time Taken:  {epoch_mins}m {epoch_secs}s"
                )


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Pretraining time {}".format(total_time_str))



if __name__ == "__main__":

    main(args)
