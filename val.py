
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import tqdm

from config import get_args
from datasets import get_dataset
from metrics import StreamSegMetrics
from net import *
from net.mod.replicate import patch_replication_callback
from utils import Denormalize
from utils import mask_overlay
from utils import make_directory, set_seed
from PIL import Image

def validate(args, logger, device, metrics, model, val_dataset, val_dataloader, val_dir=None, ret_sample_ids=None, mode='train'):
    """Do validation and return score and samples"""
    logger.info("Validation ...")
    metrics.reset()
    ret_samples = []
    ret_names = []
    
    denorm = Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    model.eval()
    with torch.no_grad():
        for step, pack in tqdm.tqdm(enumerate(val_dataloader)):
            name = pack['name']
            image = pack['img'].to(device, dtype=torch.float32)
            target = pack['target'].to(device, dtype=torch.long)
            
            output = model(image)
            pred = output.detach().max(dim=1)[1].cpu().numpy()
            target = target.cpu().numpy()
            
            metrics.update(target, pred)
            
            # Visualize samples
            img = (denorm(image[0].detach().cpu().numpy()) * 255).astype(np.uint8)
            target = val_dataset.decode_target(target[0]).transpose(2, 0, 1).astype(np.uint8)
            pred = val_dataset.decode_target(pred[0]).transpose(2, 0, 1).astype(np.uint8)
            over_lay = mask_overlay(pred, img)
            
            
            concat_img = np.concatenate((img, target, pred, over_lay), axis=2).astype(np.uint8)
            
            if mode == 'train' and ret_sample_ids is not None and step in ret_sample_ids:
                ret_samples.append(concat_img)
                ret_names.append(str(name[0]))
            elif mode == 'test':
                concat_img = Image.fromarray(concat_img.transpose(1, 2, 0))
                concat_img.save(os.path.join(val_dir, str(name[0]) + '.png'))
        
        score = metrics.get_results()
    
    return score, ret_samples, ret_names
                
                

if __name__ == '__main__':
    args = get_args()
    
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
    val_dir = os.path.join(session_dir, args.val_path)

    make_directory(val_dir)

    # Setup logger
    logger = logging.getLogger("Validation")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(os.path.join(log_dir, 'val.log'))
    logger.addHandler(file_handler)
    logger.info("Args: %s" %args)
    logger.info("Device %s" % device)

    # Setup Dataset
    _, val_dataset = get_dataset(args)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, \
        num_workers=args.num_workers, worker_init_fn=worker_init_fn)

    logger.info("Dataset: %s, Val set: %d \n" %
                (args.dataset, len(val_dataset)))
    
    # Setup model
    model = MODEL_MAP[args.based_model](args.num_classes, args.output_stride)
    model = nn.DataParallel(model, device_ids=gpu_id)
    patch_replication_callback(model)
    model.to(device)

    checkpoint_name = os.path.join(checkpoint_dir, args.checkpoint)
    checkpoint = torch.load(checkpoint_name)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # Setup metrics
    metrics = StreamSegMetrics(args.num_classes)

    val_score, ret_samples, ret_names = validate(args, logger, device, metrics, model, \
                                                 val_dataset, val_dataloader, val_dir=val_dir, \
                                                 ret_sample_ids=None, mode='test')

    logger.info(metrics.to_str(val_score))

    