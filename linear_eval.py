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

def plotly_fig2array(fig):
    fig_bytes = fig.to_image(format='png', engine='kaleido')
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf).convert("RGB")
    return np.asarray(img)

def main(args, train_loader=None, test_loader=None, model=None, tsne_visualization=False):

    if train_loader is None:
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        args.eval.num_epochs = 10

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
    if test_loader is None:
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


    if model is None:
        model = get_backbone(args.model.backbone)
        assert args.eval_from is not None
        save_dict = torch.load(args.eval_from, map_location='cpu')
        msg = model.load_state_dict({k[9:]:v for k, v in save_dict['state_dict'].items() if k.startswith('backbone.')}, strict=True)
        
        print(msg)
    else:
        model = deepcopy(model)
    classifier = nn.Linear(in_features=model.output_dim, out_features=args.eval.num_classes, bias=True).to(args.device)

    model = model.to(args.device)
    model = torch.nn.DataParallel(model)

    # if torch.cuda.device_count() > 1: classifier = torch.nn.SyncBatchNorm.convert_sync_batchnorm(classifier)
    classifier = torch.nn.DataParallel(classifier)
    # define optimizer
    optimizer = get_optimizer(
        args.eval.optimizer.name, classifier, 
        lr=args.eval.base_lr*args.eval.batch_size/256, 
        momentum=args.eval.optimizer.momentum, 
        weight_decay=args.eval.optimizer.weight_decay)

    # define lr scheduler
    lr_scheduler = LR_Scheduler(
        optimizer,
        args.eval.warmup_epochs, args.eval.warmup_lr*args.eval.batch_size/256, 
        args.eval.num_epochs, args.eval.base_lr*args.eval.batch_size/256, args.eval.final_lr*args.eval.batch_size/256, 
        len(train_loader),
    )

    loss_meter = AverageMeter(name='Loss')
    acc_meter = AverageMeter(name='Accuracy')

    # Start training
    global_progress = tqdm(range(0, args.eval.num_epochs), desc=f'Evaluating')
    for epoch in global_progress:
        loss_meter.reset()
        model.eval()
        classifier.train()
        local_progress = tqdm(train_loader, desc=f'Epoch {epoch}/{args.eval.num_epochs}', disable=True)
        
        for idx, x in enumerate(local_progress):
            images = x[0]
            labels = x[-1]

            classifier.zero_grad()
            with torch.no_grad():
                feature = model(images.to(args.device))

            preds = classifier(feature)

            loss = F.cross_entropy(preds, labels.to(args.device))

            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())
            lr = lr_scheduler.step()
            local_progress.set_postfix({'lr':lr, "loss":loss_meter.val, 'loss_avg':loss_meter.avg})

    classifier.eval()
    correct, total = 0, 0
    acc_meter.reset()
    train_features = []
    train_labels = []
    train_images = []
    for idx, x in enumerate(train_loader):
        images = x[0]
        labels = x[-1]
        with torch.no_grad():
            assert (labels == 51).sum().item() == 0
            feature = model(images.to(args.device))
            train_images.append(images)
            train_features.append(feature)
            train_labels.append(labels)
            preds = classifier(feature).argmax(dim=1)
            correct = (preds == labels.to(args.device)).sum().item()
            acc_meter.update(correct/preds.shape[0])
    train_accuracy = acc_meter.avg * 100.
    train_images = torch.cat(train_images, dim=0)
    train_features = torch.cat(train_features, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    print(f'Train Accuracy = {acc_meter.avg*100:.2f}')
    if tsne_visualization:
        train_images = train_images.cpu().numpy()
        train_features = train_features.cpu().numpy()
        train_labels = train_labels.cpu().numpy()
        train_features_embedded = TSNE(n_components=2).fit_transform(train_features)
        with imageio.get_writer('train.mp4', mode="I", fps=5) as writer:
            for idx in trange(1, len(train_images)+1):
                fig = px.scatter(x=train_features_embedded[:idx, 0], y=train_features_embedded[:idx, 1], color=[str(x) for x in train_labels[:idx]], size=[1 for _ in range(idx-1)] + [5], opacity=[(0.5 * i / float(len(train_images)+1)) + 0.5 for i in range(idx-1)], range_x=[-60, 60], range_y=[-70, 70])
                fig_np = plotly_fig2array(fig)

                attach_img = cv2.resize(train_images[idx-1].transpose((1, 2, 0)), (500, 500))
                attach_img -= attach_img.min()
                attach_img = attach_img / attach_img.max() * 255.

                out = np.concatenate((fig_np, attach_img), axis=1).astype(np.uint8)

                writer.append_data(out)
        writer.close()


    correct, total = 0, 0
    acc_meter.reset()
    test_features = []
    for idx, x in enumerate(test_loader):
        images = x[0]
        labels = x[-1]
        with torch.no_grad():
            assert (labels == 51).sum().item() == 0
            feature = model(images.to(args.device))
            test_features.append(feature)
            preds = classifier(feature).argmax(dim=1)
            correct = (preds == labels.to(args.device)).sum().item()
            acc_meter.update(correct/preds.shape[0])
    test_accuracy = acc_meter.avg * 100.
    test_features = torch.cat(test_features, dim=0)
    print(f'Test Accuracy = {acc_meter.avg*100:.2f}')

    return train_accuracy, test_accuracy, train_features, test_features




if __name__ == "__main__":
    main(args=get_args(), tsne_visualization=False)

