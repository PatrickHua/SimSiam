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
from tools import AverageMeter, PlotLogger
from datasets import get_dataset
from optimizers import get_optimizer, LR_Scheduler
from linear_eval import main as linear_eval




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
    if args.model == 'simsiam' and args.proj_layers is not None: model.projector.set_layers(args.proj_layers)
    model = torch.nn.DataParallel(model)
    if torch.cuda.device_count() > 1: model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    # define optimizer
    optimizer = get_optimizer(
        args.optimizer, model, 
        lr=args.base_lr*args.batch_size/256, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay)

    lr_scheduler = LR_Scheduler(
        optimizer,
        args.warmup_epochs, args.warmup_lr*args.batch_size/256, 
        args.num_epochs, args.base_lr*args.batch_size/256, args.final_lr*args.batch_size/256, 
        len(train_loader)
    )

    loss_meter = AverageMeter(name='Loss')
    plot_logger = PlotLogger(params=['epoch', 'lr', 'loss'])
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
            local_progress.set_postfix({'lr':lr, "loss":loss_meter.val})
            plot_logger.update({'epoch':epoch, 'lr':lr, 'loss':loss_meter.val})
        global_progress.set_postfix({"epoch":epoch, "loss_avg":loss_meter.avg})
        plot_logger.save(os.path.join(args.output_dir, 'logger.svg'))

    
        # Save checkpoint
        
    model_path = os.path.join(args.output_dir, f'{args.model}-{args.dataset}-epoch{epoch+1}.pth')
    torch.save({
        'epoch': epoch+1,
        'state_dict':model.module.state_dict(),
        # 'optimizer':optimizer.state_dict(), # will double the checkpoint file size
        'lr_scheduler':lr_scheduler,
        'args':args,
        'loss_meter':loss_meter
    }, model_path)
    print(f"Model saved to {model_path}")
        

    if args.eval_after_train:
        args.eval_from = model_path
        
        args.base_lr = 0.02
        args.weight_decay = 0
        args.momentum = 0.9
        if not args.debug: 
            args.batch_size = 4096
            args.num_epochs = 50
        # breakpoint()
        args.optimizer = 'lars'
        linear_eval(args)

if __name__ == "__main__":
    main(args=get_args())
















