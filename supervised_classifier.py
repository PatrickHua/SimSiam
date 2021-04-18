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
np.set_printoptions(threshold=sys.maxsize)

def main(args):

    args.dataset_kwargs['ordering'] = 'instance'
    train_loader = torch.utils.data.DataLoader(
        dataset=get_dataset( 
            transform=get_aug(train=True, train_classifier=False, **args.aug_kwargs), 
            train=True, 
            **args.dataset_kwargs
        ),
        batch_size=args.eval.batch_size,
        shuffle=True,
        **args.dataloader_kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs), 
            train=False,
            **args.dataset_kwargs
        ),
        batch_size=args.eval.batch_size,
        shuffle=False,
        **args.dataloader_kwargs
    )


    model = get_backbone(args.model.backbone)

    classifier = nn.Sequential(model, nn.Linear(in_features=model.output_dim, out_features=args.eval.num_classes, bias=True))

    # if torch.cuda.device_count() > 1: classifier = torch.nn.SyncBatchNorm.convert_sync_batchnorm(classifier)
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

    # Start training
    global_progress = tqdm(range(0, args.eval.num_epochs), desc=f'Evaluating')
    for epoch in global_progress:
        loss_meter.reset()
        classifier.train()
        local_progress = tqdm(train_loader, desc=f'Epoch {epoch}/{args.eval.num_epochs}', disable=True)
        
        total_losses = 0.
        for idx, tup in enumerate(local_progress):
            images = tup[0]
            labels = tup[-1]

            classifier.zero_grad()

            preds = classifier(images.to(args.device))

            loss = F.cross_entropy(preds, labels.to(args.device))

            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())
            total_losses += loss.item()
            lr = lr_scheduler.step()
            local_progress.set_postfix({'lr':lr, "loss":loss_meter.val, 'loss_avg':loss_meter.avg})
        if (epoch+1) % 10 == 0:
            print("Epoch Loss:", total_losses / len(local_progress))

    classifier.eval()
    correct, total = 0, 0
    acc_meter.reset()
    train_features = []
    train_labels = []
    train_images = []
    for idx, tup in enumerate(train_loader):
        images = tup[0]
        labels = tup[-1]
        with torch.no_grad():
            assert (labels == 51).sum().item() == 0
            feature = model(images.to(args.device))
            train_images.append(images)
            train_features.append(feature)
            train_labels.append(labels)
            preds = classifier(images.to(args.device)).argmax(dim=1)
            correct = (preds == labels.to(args.device)).sum().item()
            acc_meter.update(correct/preds.shape[0])
    train_accuracy = acc_meter.avg * 100.
    train_images = torch.cat(train_images, dim=0)
    train_features = torch.cat(train_features, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    print(f'Train Accuracy = {acc_meter.avg*100:.2f}')

    correct, total = 0, 0
    acc_meter.reset()
    test_features = []
    test_predictions = []
    test_labels = []
    for idx, (images, labels) in enumerate(test_loader):
        with torch.no_grad():
            assert (labels == 51).sum().item() == 0
            feature = model(images.to(args.device))
            test_features.append(feature)
            preds = classifier(images.to(args.device)).argmax(dim=1)
            correct = (preds == labels.to(args.device)).sum().item()
            test_predictions.append(preds)
            test_labels.append(labels)
            acc_meter.update(correct/preds.shape[0])
    test_accuracy = acc_meter.avg * 100.
    test_features = torch.cat(test_features, dim=0)
    test_predictions = torch.cat(test_predictions, dim=0).cpu().numpy()
    test_labels = torch.cat(test_labels, dim=0).cpu().numpy()
    print(f'Test Accuracy = {acc_meter.avg*100:.2f}')
    conf_mat = confusion_matrix(test_labels, test_predictions, normalize='true')
    fig = px.imshow(conf_mat)
    fig.write_image("test_confusion.png")


    return train_accuracy, test_accuracy, train_features, test_features




if __name__ == "__main__":
    main(args=get_args())

