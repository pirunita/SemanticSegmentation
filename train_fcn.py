import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
import tqdm

from PIL import Image

from config import get_args
from datasets import get_dataset
from metrics import StreamSegMetrics
from net import *
from net.mod.replicate import patch_replication_callback
from utils import PolyLR, Visualizer
from utils import make_directory, save_checkpoint, set_seed
from val import validate

# Setup logger
logger = logging.getLogger("FCN")
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)


def train(args):
    # Setup random seed
    if args.random_seed >= 0:
        set_seed(args.random_seed)
    if args.random_seed >= 0: 
        worker_init_fn = np.random.seed(args.random_seed)
    else:
        worker_init_fn = None
    
    # Set Device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_id = [int(s) for s in args.gpu_id.split(',')]


    if args.dataset.lower() == 'voc':
        args.num_classes = 21

    # Setup path
    root_dir = args.name
    session_dir = os.path.join(root_dir, str(args.session))
    log_dir = os.path.join(session_dir, args.log_path)
    checkpoint_dir = os.path.join(session_dir, args.checkpoint_path)
    training_val_dir = os.path.join(session_dir, args.training_val_path)

    make_directory(root_dir, session_dir, log_dir, checkpoint_dir, training_val_dir)

    # Setup logger
    file_handler = logging.FileHandler(os.path.join(log_dir, 'train.log'))
    logger.addHandler(file_handler)
    logger.info("Args: %s" %args)
    logger.info("Device %s" % device)

    # Setup Dataset
    train_dataset, val_dataset = get_dataset(args)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, \
        num_workers=args.num_workers, worker_init_fn=worker_init_fn)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, \
        num_workers=args.num_workers, worker_init_fn=worker_init_fn)
    
    logger.info("Dataset: %s, Train set: %d , Train dataloader: %d, Val set: %d \n" %
                (args.dataset, len(train_dataset), len(train_dataloader),len(val_dataset)))

    # Setup model & Criterion & Optimizer & Scheduler
    # model_map in net/__init__.py
    model = MODEL_MAP[args.based_model](args.num_classes, args.output_stride)
    model = nn.DataParallel(model, device_ids=gpu_id)
    patch_replication_callback(model)
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    optimizer = torch.optim.SGD(params=[
        {'params': model.module.backbone.parameters(), 'lr': 0.1*args.lr},
        {'params': model.module.classifier.parameters(), 'lr': args.lr},
    ], lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    total_epoch = args.total_epoch

    if args.lr_policy == 'poly':
        total_step = len(train_dataloader) * total_epoch
        scheduler = PolyLR(optimizer, total_step, power=0.9)
    
    # Setup Visualizer & Metric
    vis = Visualizer(port=args.vis_port, env=args.vis_env) if args.enable_vis else None
    vis_sample_id = np.random.randint(0, len(val_dataloader), args.vis_num_samples, np.int32) if args.enable_vis else None
    metrics = StreamSegMetrics(args.num_classes)

    # Initialize value
    interval_loss = 0.0
    best_score = 0.0
    current_epoch = 0
    current_step = 0
    print_step = len(train_dataloader) // 10
    for epoch in tqdm.tqdm(range(total_epoch), total=total_epoch):
        model.train()
        current_epoch += 1

        for step, pack in tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            current_step += 1
            images = pack['img'].to(device, dtype=torch.float32)
            targets = pack['target'].to(device, dtype=torch.long)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss

            if vis is not None:
                vis.vis_scalar('Loss', current_step, np_loss)
            if step % print_step == 0:
                logger.info("[Session %d] [Epoch %d/%d, Step %d/%d] [loss %.7f] [Lr: %.4f] " %(
                    args.session, current_epoch, total_epoch, step, len(train_dataloader),
                    interval_loss / print_step, optimizer.param_groups[0]['lr']))
                interval_loss = 0.0

        val_score, ret_samples, ret_names = validate(args, logger, device, metrics, model, \
                                                     val_dataset, val_dataloader, val_dir=None, \
                                                     ret_sample_ids=vis_sample_id, mode='train')
        
        logger.info(metrics.to_str(val_score))
        if val_score['Mean IoU'] > best_score:
            best_score = val_score['Mean IoU']
            save_checkpoint(state="best", path=checkpoint_dir, dataset=args.dataset, epoch=current_epoch, \
                            model=model, optimizer=optimizer, scheduler=scheduler, best_score=best_score)
            
        else:
            save_checkpoint(state="latest", path=checkpoint_dir, dataset=args.dataset, epoch=current_epoch, \
                            model=model, optimizer=optimizer, scheduler=scheduler, best_score=best_score)

        if vis is not None:
            vis.vis_scalar("[Val] Overall Acc", current_epoch, val_score['Overall Acc'])
            vis.vis_scalar("[Val] Mean IoU", current_epoch, val_score['Mean IoU'])
            vis.vis_table("[Val] Class IoU", val_score['Class IoU'])
            
            for sample, name in zip(ret_samples, ret_names):
                vis.vis_image('Sample %s' %name, sample)
                concat_path = os.path.join(training_val_dir, name)
                if not os.path.exists(concat_path):
                    make_directory(concat_path)
                
                concat_img = Image.fromarray(sample.transpose(1, 2, 0))
                concat_img.save(os.path.join(concat_path, '%d_epoch_%s.png' %(current_epoch, name)))

        model.train()

if __name__ == '__main__':
    args = get_args()
    train(args)