import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
from tqdm import tqdm
from configs import get_args
from augmentations import get_aug
from models import get_model, get_backbone
from tools import AverageMeter
from datasets import get_dataset
from optimizers import get_optimizer

def main(args):

    train_set = get_dataset(
        args.dataset, 
        args.data_dir, 
        transform=get_aug(args.model, args.image_size, train=False, train_classifier=True), 
        train=True, 
        download=args.download, # default is False
        debug_subset_size=args.batch_size if args.debug else None
    )
    test_set = get_dataset(
        args.dataset, 
        args.data_dir, 
        transform=get_aug(args.model, args.image_size, train=False, train_classifier=False), 
        train=False, 
        download=args.download, # default is False
        debug_subset_size=args.batch_size if args.debug else None
    )


    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    model = get_backbone(args.backbone)
    classifier = nn.Linear(in_features=model.output_dim, out_features=10, bias=True).to(args.device)

    assert args.eval_from is not None
    save_dict = torch.load(args.eval_from, map_location='cpu')
    msg = model.load_state_dict({k[9:]:v for k, v in save_dict['state_dict'].items() if k.startswith('backbone.')}, strict=True)
    
    # print(msg)
    model = model.to(args.device)
    model = torch.nn.DataParallel(model)

    # if torch.cuda.device_count() > 1: classifier = torch.nn.SyncBatchNorm.convert_sync_batchnorm(classifier)
    classifier = torch.nn.DataParallel(classifier)
    # define optimizer
    optimizer = get_optimizer(
        args.optimizer, classifier, 
        lr=args.base_lr*args.batch_size/256, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay)

    # TODO: linear lr warm up for byol simclr swav
    # args.warm_up_epochs

    # define lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.num_epochs, eta_min=0)

    loss_meter = AverageMeter(name='Loss')
    acc_meter = AverageMeter(name='Accuracy')

    # Start training
    global_progress = tqdm(range(0, args.num_epochs), desc=f'Evaluating')
    for epoch in global_progress:
        loss_meter.reset()
        model.eval()
        classifier.train()
        local_progress = tqdm(train_loader, desc=f'Epoch {epoch}/{args.num_epochs}', disable=args.hide_progress)
        
        for idx, (images, labels) in enumerate(local_progress):

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
        

        if args.head_tail_accuracy and epoch != 0 and epoch != args.num_epochs: continue

        local_progress=tqdm(test_loader, desc=f'Test {epoch}/{args.num_epochs}', disable=args.hide_progress)
        classifier.eval()
        correct, total = 0, 0
        acc_meter.reset()
        for idx, (images, labels) in enumerate(local_progress):
            with torch.no_grad():
                feature = model(images.to(args.device))
                preds = classifier(feature).argmax(dim=1)
                correct = (preds == labels.to(args.device)).sum().item()
                acc_meter.update(correct/preds.shape[0])
                local_progress.set_postfix({'accuracy': acc_meter.avg})
        
        global_progress.set_postfix({"epoch":epoch, 'accuracy':acc_meter.avg*100})




if __name__ == "__main__":
    main(args=get_args())
















