import argparse
import os
import time
import numpy as np
import random
import sys

from torch.nn.modules.batchnorm import _BatchNorm
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, required
import math

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from utils import *

# Configuration dictionary
config = {
    "device": 0,
    "seed": 1,
    "datasets": "CIFAR100",
    "model": "resnet18",      # resnet18, VGG16BN, WideResNet28x10
    "schedule": "cosine",     # step, cosine
    "wd": 0.001,
    "epochs": 200,
    "batch_size": 128,
    "lr": 0.05,
    "rho": 0.2,
    "sigma": 1,
    "lmbda": 0.6,
    "optimizer": "FriendlySAM",  # FriendlySAM, SAM, etc.
    "print_freq": 200,
    "save_dir": "results/FriendlySAM/CIFAR100/resnet18/...",
    "log_dir": "results/FriendlySAM/CIFAR100/resnet18/...",
    "resume": "",
    "evaluate": False,
    "wandb": False,
    "half": False,
    "randomseed": 1,
    "cutout": True,
    "noise_ratio": 0.5,
    "img_size": 224,
    "drop": 0.0,
    "drop_path": 0.1,
    "drop_block": None,
    "patch": 4,
    "dimhead": 512,
    "convkernel": 8,
}

best_prec1 = 0

train_loss = []
train_err = []
post_train_loss = []
post_train_err = []
ori_train_loss = []
ori_train_err = []
test_loss = []
test_err = []
arr_time = []

p0 = None

def disable_running_stats(model):
    # Disable running stats for SAM
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0
    model.apply(_disable)

def enable_running_stats(model):
    # Enable running stats for SAM
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum
    model.apply(_enable)

def _cosine_annealing(step, total_steps, lr_max, lr_min):
    # Cosine annealing LR
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

def get_cosine_annealing_scheduler(optimizer, epochs, steps_per_epoch, base_lr):
    # Scheduler for cosine annealing
    lr_min = 0.0
    total_steps = epochs * steps_per_epoch
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _cosine_annealing(step, total_steps, 1, lr_min / base_lr)
    )

def main():
    global best_prec1, p0
    global train_loss, train_err, post_train_loss, post_train_err, test_loss, test_err, arr_time
    args = config

    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args["device"])
    set_seed(args["seed"])

    # Make dirs
    if not os.path.exists(args["save_dir"]):
        os.makedirs(args["save_dir"])
    if not os.path.exists(args["log_dir"]):
        os.makedirs(args["log_dir"])

    # Redirect stdout
    sys.stdout = Logger(os.path.join(args["log_dir"], "log.txt"))
    print("Save dir:", args["save_dir"])
    print("Log dir:", args["log_dir"])

    # Model
    model = get_model_from_name(args["model"], datasets=args["datasets"])
    model = model.cuda()

    # Resume
    if args["resume"]:
        if os.path.isfile(args["resume"]):
            print("=> loading checkpoint '{}'".format(args["resume"]))
            checkpoint = torch.load(args["resume"])
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args["resume"]))

    cudnn.benchmark = True

    # Dataloaders
    if args["cutout"]:
        train_loader, val_loader = get_datasets_cutout(args)
    else:
        train_loader, val_loader = get_normal_datasets(args)

    # Loss
    criterion = nn.CrossEntropyLoss().cuda()

    if args["half"]:
        model.half()
        criterion.half()

    # Optimizer
    print("Optimizer:", args["optimizer"])
    if args["optimizer"] == "SAM":
        base_optimizer = torch.optim.SGD
        optimizer = SAM(model.parameters(), base_optimizer, rho=args["rho"],
                        adaptive=0, lr=args["lr"], momentum=0.9, weight_decay=args["wd"])
    elif args["optimizer"] == "FriendlySAM":
        base_optimizer = torch.optim.SGD
        optimizer = FriendlySAM(model.parameters(), base_optimizer,
                                rho=args["rho"], sigma=args["sigma"], lmbda=args["lmbda"],
                                adaptive=0, lr=args["lr"], momentum=0.9, weight_decay=args["wd"])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args["lr"],
                                    momentum=0.9, weight_decay=args["wd"])

    # LR scheduler
    if args["schedule"] == "step":
        if hasattr(optimizer, 'base_optimizer'):
            base_opt = optimizer.base_optimizer
        else:
            base_opt = optimizer
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(base_opt, milestones=[60,120,160], gamma=0.2)
    elif args["schedule"] == "cosine":
        lr_scheduler = get_cosine_annealing_scheduler(optimizer, args["epochs"], len(train_loader), args["lr"])
    else:
        lr_scheduler = None

    # Evaluate only
    if args["evaluate"]:
        validate(val_loader, model, criterion, args)
        return

    # Training
    is_best = False
    print("Start training...")
    torch.save(model.state_dict(), os.path.join(args["save_dir"], "initial.pt"))
    p0 = get_model_param_vec(model)

    for epoch in range(args["epochs"]):
        if lr_scheduler and isinstance(lr_scheduler, torch.optim.lr_scheduler.MultiStepLR):
            print("Current LR: {:.5e}".format(lr_scheduler.get_last_lr()[0]))
        elif lr_scheduler:
            print("Current LR: {:.5e}".format(lr_scheduler.optimizer.param_groups[0]['lr']))

        train(train_loader, model, criterion, optimizer, lr_scheduler, epoch, args)
        prec1 = validate(val_loader, model, criterion, args)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        save_checkpoint(
            {
                "state_dict": model.state_dict(),
                "best_prec1": best_prec1
            },
            is_best,
            filename=os.path.join(args["save_dir"], "model.th")
        )

    print("Finished Training")
    print("Best top-1 accuracy:", best_prec1)
    print("Train loss:", train_loss)
    print("Train err:", train_err)
    print("Test loss:", test_loss)
    print("Test err:", test_err)
    print("Time array:", arr_time)

def train(train_loader, model, criterion, optimizer, lr_scheduler, epoch, args):
    global train_loss, train_err, post_train_loss, post_train_err, ori_train_loss, ori_train_err, arr_time

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    total_loss = 0
    total_err = 0
    ori_total_loss = 0
    ori_total_err = 0

    end = time.time()

    for i, (input_data, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input_data.cuda()
        target_var = target
        if args["half"]:
            input_var = input_var.half()

        enable_running_stats(model)
        predictions = model(input_var)
        loss = criterion(predictions, target_var)
        loss.mean().backward()
        optimizer.first_step(zero_grad=True)

        disable_running_stats(model)
        output_adv = model(input_var)
        loss_adv = criterion(output_adv, target_var)
        loss_adv.mean().backward()
        optimizer.second_step(zero_grad=True)

        # Step scheduler if not MultiStep
        if lr_scheduler and not isinstance(lr_scheduler, torch.optim.lr_scheduler.MultiStepLR):
            lr_scheduler.step()

        output = predictions.float()
        loss = loss.float()

        batch_sz = input_var.shape[0]
        total_loss += loss.item() * batch_sz
        total_err += (output.max(dim=1)[1] != target_var).sum().item()

        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), batch_sz)
        top1.update(prec1.item(), batch_sz)

        batch_time.update(time.time() - end)
        end = time.time()

        ori_total_loss += loss_adv.item() * batch_sz
        ori_total_err += (output_adv.max(dim=1)[1] != target_var).sum().item()

        if i % args["print_freq"] == 0 or i == len(train_loader) - 1:
            print("Epoch: [{0}][{1}/{2}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                  "Data {data_time.val:.3f} ({data_time.avg:.3f})\tLoss {loss.val:.4f} ({loss.avg:.4f})\t"
                  "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))

    print("Total time for epoch [{0}]: {1:.3f}s".format(epoch, batch_time.sum))
    dataset_size = len(train_loader.dataset)
    train_loss.append(total_loss / dataset_size)
    train_err.append(total_err / dataset_size)
    ori_train_loss.append(ori_total_loss / dataset_size)
    ori_train_err.append(ori_total_err / dataset_size)
    arr_time.append(batch_time.sum)

def validate(val_loader, model, criterion, args):
    global test_err, test_loss
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    total_loss = 0
    total_err = 0

    model.eval()
    end = time.time()

    with torch.no_grad():
        for i, (input_data, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input_data.cuda()
            target_var = target
            if args["half"]:
                input_var = input_var.half()

            output = model(input_var)
            loss = criterion(output, target_var)
            output = output.float()
            loss = loss.float()

            batch_sz = input_var.shape[0]
            total_loss += loss.item() * batch_sz
            total_err += (output.max(dim=1)[1] != target_var).sum().item()

            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), batch_sz)
            top1.update(prec1.item(), batch_sz)

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args["print_freq"] == 0:
                print("Test: [{0}/{1}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                      "Loss {loss.val:.4f} ({loss.avg:.4f})\tPrec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                       i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1))

    print(" * Prec@1 {top1.avg:.3f}".format(top1=top1))

    dataset_size = len(val_loader.dataset)
    test_loss.append(total_loss / dataset_size)
    test_err.append(total_err / dataset_size)
    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    # Save model
    torch.save(state, filename)

class AverageMeter(object):
    # AverageMeter to track stats
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    # Compute accuracy@k
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == "__main__":
    main()
