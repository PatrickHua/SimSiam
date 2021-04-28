import os
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
from tqdm import tqdm, trange
from arguments import get_args
from augmentations import get_aug
from models import get_model, get_backbone
from tools import AverageMeter
from datasets import get_dataset
from optimizers import get_optimizer, LR_Scheduler
from sklearn.manifold import TSNE
import plotly.express as px
import io
from PIL import Image
import imageio
import numpy as np
import cv2
import random
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import sys
import wandb
import pandas as pd
np.set_printoptions(threshold=sys.maxsize)

def iterate_and_write(dataloader, path):
    img_idx = 0
    for x, _, y in tqdm(dataloader):
        label_idx = y.item()
        img = (x[0, :, :, :].numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        img = Image.fromarray(img)

        imgpath = os.path.join(path, "label" + str(label_idx))
        os.makedirs(imgpath, exist_ok=True)
        img.save(os.path.join(imgpath, "img" + str(img_idx) + ".bmp"))

        img_idx += 1

def write_images(args):
    args.dataset_kwargs['ordering'] = 'instance'
    convert_train = True
    if convert_train:
        dataloader = torch.utils.data.DataLoader(
            dataset=get_dataset( 
                transform=torchvision.transforms.Resize(64),
                train=True, 
                **args.dataset_kwargs
            ),
            batch_size=1,
            shuffle=False,
            **args.dataloader_kwargs
        )

        path = os.path.join("..", "ucfimages64x64-train")

    else:
        datalaoder = torch.utils.data.DataLoader(
            dataset=get_dataset(
                transform=torchvision.transforms.Resize(64),
                train=False,
                **args.dataset_kwargs
            ),
            batch_size=1,
            shuffle=False,
            **args.dataloader_kwargs
        )

        path = os.path.join("..", "ucfimages64x64-test")
    iterate_and_write(dataloader, path)

def calc_accuracy(classifier, dataloader, device):
    with torch.no_grad():
        classifier.eval()
        acc_meter = AverageMeter('Accuracy')
        correct, total = 0, 0
        acc_meter.reset()
        all_predictions = []
        all_labels = []
        for idx, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            images = data[0]
            labels = data[-1]
            with torch.no_grad():
                preds = classifier(images.to(device, non_blocking=True)).argmax(dim=1)
                correct = (preds == labels.to(device, non_blocking=True)).sum().item()
                all_predictions.append(preds)
                all_labels.append(labels)
                acc_meter.update(correct/preds.shape[0])
        accuracy = acc_meter.avg
        all_predictions = torch.cat(all_predictions, dim=0).cpu().numpy()
        all_labels = torch.cat(all_labels, dim=0).cpu().numpy()

    return accuracy, all_labels, all_predictions

def main(args):
    if args.wandb:
        wandb_config = pd.json_normalize(vars(args), sep='_')
        wandb_config = wandb_config.to_dict(orient='records')[0]
        wandb.init(project='simsiam', config=wandb_config)

    args.dataset_kwargs['ordering'] = 'instance'
    train_aug = get_aug(train=True, train_classifier=False, **args.aug_kwargs)
    train_loader = torch.utils.data.DataLoader(
        dataset=get_dataset( 
            transform=train_aug, 
            train=True, 
            **args.dataset_kwargs
        ),
        batch_size=args.eval.batch_size,
        shuffle=True,
        **args.dataloader_kwargs
    )
    test_aug = get_aug(train=False, train_classifier=False, **args.aug_kwargs)
    test_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=test_aug, 
            train=False,
            **args.dataset_kwargs
        ),
        batch_size=args.eval.batch_size,
        shuffle=True,
        **args.dataloader_kwargs
    )

    model = get_backbone(args.model.backbone)

    classifier = nn.Sequential(model, nn.Linear(in_features=model.output_dim, out_features=args.eval.num_classes, bias=True))

    classifier = classifier.to(args.device)
    classifier = torch.nn.DataParallel(classifier)
    # define optimizer
    optimizer = get_optimizer(
        args.eval.optimizer.name, classifier, 
        lr=0.03*args.eval.batch_size/256, 
        momentum=args.eval.optimizer.momentum, 
        weight_decay=args.eval.optimizer.weight_decay)

    # define lr scheduler
    lr_scheduler = LR_Scheduler(
        optimizer,
        args.eval.warmup_epochs, args.eval.warmup_lr*args.eval.batch_size/256, 
        args.eval.num_epochs, 0.03*args.eval.batch_size/256, args.eval.final_lr*args.eval.batch_size/256, 
        len(train_loader),
    )

    loss_meter = AverageMeter(name='Loss')
    acc_meter = AverageMeter(name='Accuracy')

    if args.wandb:
        wandb.watch(classifier)

    # Start training
    global_progress = tqdm(range(0, args.eval.num_epochs), desc=f'Evaluating')
    train_accuracy = 0.
    test_accuracy = 0.
    for epoch in global_progress:
        classifier.train()
        loss_meter.reset()
        classifier.train()
        local_progress = tqdm(train_loader, desc=f'Epoch {epoch}/{args.eval.num_epochs}', disable=True)
        
        for idx, tup in enumerate(local_progress):
            images = tup[0]
            labels = tup[-1]

            classifier.zero_grad()

            preds = classifier(images.to(args.device, non_blocking=True))

            loss = F.cross_entropy(preds, labels.to(args.device, non_blocking=True))

            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())
            lr = lr_scheduler.step()
            local_progress.set_postfix({'lr':lr, "loss":loss_meter.val, 'loss_avg':loss_meter.avg})
        if (epoch+1) % 5 == 0:
            train_accuracy, _, _ = calc_accuracy(classifier, train_loader, args.device)
            test_accuracy, _, _ = calc_accuracy(classifier, test_loader, args.device)
            print("Epoch Loss:", loss_meter.avg, "train acc:", train_accuracy * 100, "test acc:", test_accuracy * 100)

        if args.wandb:
            epoch_dict = {"Epoch": epoch, "Train Accuracy": train_accuracy, "Test Accuracy": test_accuracy, "Loss": loss_meter.avg}
            wandb.log(epoch_dict)

    train_accuracy, train_labels, train_predictions = calc_accuracy(classifier, train_loader, args.device)
    print(f'Train Accuracy = {acc_meter.avg*100:.2f}')

    test_accuracy, test_labels, test_predictions = calc_accuracy(classifier, test_loader, args.device)
    print(f'Test Accuracy = {test_accuracy*100:.2f}')


    return train_accuracy, test_accuracy, train_features, test_features




if __name__ == "__main__":
    main(args=get_args())
    # write_images(args=get_args())

