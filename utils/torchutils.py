import os
import random
import torch
import numpy as np

from torchvision.transforms.functional import normalize

class Denormalize(object):
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean/std
        self._std = 1/std
    
    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        return normalize(tensor, self._mean, self._std)


def save_checkpoint(state, path, dataset, epoch, model, optimizer, scheduler, best_score):
    """Save current model"""
    filepath = os.path.join(path, '{}_{}_{}.pth'.format(state, dataset, epoch))
    
    try:
        torch.save({
            "current_epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score
        }, filepath)
    except:
        raise Exception

def set_seed(seed):
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
