import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import numpy as np
from tqdm import tqdm
from arguments import get_args
from augmentations import get_aug
from models import get_model
from tools import AverageMeter, knn_monitor, Logger, file_exist_check
from datasets import get_dataset
from optimizers import get_optimizer, LR_Scheduler
from linear_eval import main as linear_eval
from datetime import datetime
import sys
import wandb
import pandas as pd

def main(device, args):

    if args.dataset_kwargs['ordering'] == 'iid' and not args.no_augmentation:
        train_loader = torch.utils.data.DataLoader(
            dataset=get_dataset(
                transform=get_aug(train=True, **args.aug_kwargs), 
                train=True,
                **args.dataset_kwargs),
            shuffle=False,
            batch_size=args.train.batch_size,
            **args.dataloader_kwargs
        )
    elif args.dataset_kwargs['ordering'] == 'iid' and args.no_augmentation:
        print("NO AUGMENTATION IID", flush=True)
        train_loader = torch.utils.data.DataLoader(
            dataset=get_dataset(
                transform=get_aug(train=False, train_classifier=True, **args.aug_kwargs), 
                train=True,
                **args.dataset_kwargs),
            shuffle=True,
            batch_size=args.train.batch_size,
            **args.dataloader_kwargs
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset=get_dataset(
                transform=get_aug(train=False, train_classifier=True, **args.aug_kwargs), 
                # transform=get_aug(train=True, **args.aug_kwargs), 
                train=True,
                **args.dataset_kwargs),
            shuffle=False,
            batch_size=args.train.batch_size,
            **args.dataloader_kwargs
        )
    memory_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs), 
            train=True,
            **args.dataset_kwargs),
        shuffle=False,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=get_dataset( 
            transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs), 
            train=False,
            **args.dataset_kwargs),
        shuffle=False,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs
    )

    # define model
    model = get_model(args.model).to(device)
    model = torch.nn.DataParallel(model)

    if args.wandb:
        wandb.watch(model)

    # define optimizer
    optimizer = get_optimizer(
        args.train.optimizer.name, model, 
        lr=args.train.base_lr*args.train.batch_size/256, 
        momentum=args.train.optimizer.momentum,
        weight_decay=args.train.optimizer.weight_decay)

    lr_scheduler = LR_Scheduler(
        optimizer,
        args.train.warmup_epochs, args.train.warmup_lr*args.train.batch_size/256, 
        args.train.num_epochs, args.train.base_lr*args.train.batch_size/256, args.train.final_lr*args.train.batch_size/256, 
        len(train_loader),
        constant_predictor_lr=True # see the end of section 4.2 predictor
    )

    logger = Logger(tensorboard=args.logger.tensorboard, matplotlib=args.logger.matplotlib, log_dir=args.log_dir)
    accuracy = 0 
    # Start training
    if args.train.knn_monitor: 
        train_accuracy = knn_monitor(model.module.backbone, memory_loader, memory_loader, device, k=min(args.train.knn_k, len(memory_loader.dataset)), hide_progress=args.hide_progress) 
        test_accuracy = knn_monitor(model.module.backbone, memory_loader, test_loader, device, k=min(args.train.knn_k, len(memory_loader.dataset)), hide_progress=args.hide_progress) 
        print("before training (train, test) accuracy", train_accuracy, test_accuracy)
    
    global_progress = tqdm(range(0, args.train.stop_at_epoch), desc=f'Training')
    for epoch in global_progress:
        model.train()

        batch_loss = 0.
        batch_updates = 0
        
        local_progress=tqdm(train_loader, desc=f'Epoch {epoch}/{args.train.num_epochs}', disable=args.hide_progress)
        for idx, (images, labels) in enumerate(local_progress):
            if len(images) == 2:
                print("IID", flush=True)
                images1 = images[0]
                images2 = images[1]

                model.zero_grad()
                data_dict = model.forward(images1.to(device, non_blocking=True), images2.to(device, non_blocking=True))
                loss = data_dict['loss'].mean() # ddp
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                data_dict.update({'lr':lr_scheduler.get_lr()})
                
                local_progress.set_postfix(data_dict)
                logger.update_scalers(data_dict)

                batch_loss += loss.item()
                batch_updates += 1

            elif args.class_awareness:
                print("N OFFSET IS", args.n_offset, flush=True)
                loss = 0.
                labels_set = {l.item() for l in labels}
                for l in labels_set:
                    images_l = images[labels == l]
                    if len(images_l) < args.n_offset + 1:
                        continue
                    images1 = torch.roll(images_l, args.n_offset)
                    images2 = images_l


                    data_dict = model.forward(images1.to(device, non_blocking=True), images2.to(device, non_blocking=True))
                    loss += (float(len(images_l)) / float(len(images))) * data_dict['loss'].mean() # ddp

                model.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                data_dict.update({'lr':lr_scheduler.get_lr()})
                
                local_progress.set_postfix(data_dict)
                logger.update_scalers(data_dict)

                batch_loss += loss.item()
                batch_updates += 1

            else:
                print("NO CLASs AWARENESS", flush=True)
                images1 = torch.roll(images, args.n_offset)
                images2 = images

                model.zero_grad()
                data_dict = model.forward(images1.to(device, non_blocking=True), images2.to(device, non_blocking=True))
                loss = data_dict['loss'].mean() # ddp
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                data_dict.update({'lr':lr_scheduler.get_lr()})
                
                local_progress.set_postfix(data_dict)
                logger.update_scalers(data_dict)

                batch_loss += loss.item()
                batch_updates += 1

        if args.train.knn_monitor and epoch % args.train.knn_interval == 0: 
            train_accuracy, train_features = knn_monitor(model.module.backbone, memory_loader, memory_loader, device, k=min(args.train.knn_k, len(memory_loader.dataset)), hide_progress=args.hide_progress) 
            test_accuracy, test_features = knn_monitor(model.module.backbone, memory_loader, test_loader, device, k=min(args.train.knn_k, len(memory_loader.dataset)), hide_progress=args.hide_progress) 
        
        epoch_dict = {"train_accuracy": train_accuracy, "test_accuracy": test_accuracy, "batch_loss": batch_loss / batch_updates, "train_feature_std": torch.std(train_features, dim=0).mean().item(), "test_feature_std": torch.std(test_features, dim=0).mean().item()}
        if args.wandb:
            wandb.log(epoch_dict)

        global_progress.set_postfix(epoch_dict)
        logger.update_scalers(epoch_dict)
    
    # Save checkpoint
    model_path = os.path.join(args.ckpt_dir, f"{args.name}_{datetime.now().strftime('%m%d%H%M%S')}.pth") # datetime.now().strftime('%Y%m%d_%H%M%S')
    torch.save({
        'epoch': epoch+1,
        'state_dict':model.module.state_dict()
    }, model_path)
    print(f"Model saved to {model_path}")
    with open(os.path.join(args.log_dir, f"checkpoint_path.txt"), 'w+') as f:
        f.write(f'{model_path}')

    if args.eval is not False:
        args.eval_from = model_path
        linear_eval(args)


if __name__ == "__main__":
    args = get_args()

    if args.wandb:
        wandb_config = pd.json_normalize(vars(args), sep='_')
        wandb_config = wandb_config.to_dict(orient='records')[0]
        wandb.init(project='simsiam', config=wandb_config)

    print("Using device", args.device)
    assert(args.device == 'cuda')

    main(device=args.device, args=args)

    completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')



    os.rename(args.log_dir, completed_log_dir)
    print(f'Log file has been saved to {completed_log_dir}')
