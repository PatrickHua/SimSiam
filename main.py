import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import numpy as np
from tqdm import tqdm
from configs import get_args
from augmentations import get_aug
from models import get_model
from tools import AverageMeter
from datasets import get_dataset
from optimizers import get_optimizer, LR_Scheduler




def main(args):

    train_set = get_dataset(
        args.dataset, 
        args.data_dir, 
        transform=get_aug(args.model, args.image_size, True), 
        train=True, 
        download=args.download, # default is False
        debug_subset_size=args.batch_size if args.debug else None # run one batch if debug
    )
    


    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # define model
    model = get_model(args.model, args.backbone).to(args.device)
    
    model = torch.nn.DataParallel(model)
    if torch.cuda.device_count() > 1: model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    # define optimizer
    optimizer = get_optimizer(
        args.optimizer, model, 
        lr=args.base_lr*args.batch_size/256, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay)

    # TODO: linear lr warm up for byol simclr swav
    # args.warm_up_epochs

    # define lr scheduler
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, args.num_epochs, eta_min=0)
    lr_scheduler = LR_Scheduler(
        optimizer,
        args.warmup_epochs, args.warmup_lr, 
        args.num_epochs, args.base_lr, args.final_lr, 
        len(train_loader)
    )

    # args.warmup_epochs

    loss_meter = AverageMeter(name='Loss')

    # Start training
    global_progress = tqdm(range(0, args.stop_at_epoch), desc=f'Training')
    for epoch in global_progress:
        loss_meter.reset()
        model.train()

        local_progress=tqdm(train_loader, desc=f'Epoch {epoch}/{args.num_epochs}', disable=args.hide_progress)
        for idx, ((images1, images2), labels) in enumerate(local_progress):

            model.zero_grad()
            loss = model.forward(images1.to(args.device), images2.to(args.device))
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())
            lr = lr_scheduler.step()
            local_progress.set_postfix({'lr':lr, "loss":loss_meter.val, 'loss_avg':loss_meter.avg})
        global_progress.set_postfix({"epoch":epoch, "loss":loss_meter.avg})



        # Save checkpoint
        os.makedirs(args.output_dir, exist_ok=True)
        model_path = os.path.join(args.output_dir, f'{args.model}-{args.dataset}-epoch{epoch+1}.pth')
        torch.save({
            'epoch': epoch+1,
            'state_dict':model.module.state_dict(),
            # 'optimizer':optimizer.state_dict(), # will double the checkpoint file size
            'lr_scheduler':lr_scheduler.state_dict(),
            'args':args,
            'loss_meter':loss_meter
        }, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main(args=get_args())
















