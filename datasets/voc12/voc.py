import os

import numpy as np
import torch.utils.data as data

from PIL import Image

from utils import set_seed

IMG_FOLDER_NAME = 'JPEGImages'
ANNOT_FOLDER_NAME = 'Annotations'
TARGET_FOLDER_NAME = 'SegmentationClass'

def decode_int_filename(int_filename):
    s = str(int(int_filename))
    return s[:4] + '_' + s[4:]
    
def decode_gtlabels(masks):
    stacked_img = np.stack((masks,)*3, axis=-1)
    return 255 * stacked_img

def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)
    
    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3
        
        cmap[i] = np.array([r, g, b])
    
    cmap = cmap/255 if normalized else cmap
    return cmap


class VOCSegmentation(data.Dataset):
    cmap = voc_cmap()
    """ Pascal VOC traininset for Weakly Supervised Semantic Segmentation
    Args:
        args        (string)    : Argument
        transform   (callable) : 
    """
    def __init__(self, args, voc12_root, img_name_list_path, transform, seed=-1):
        if seed >= 0:
            set_seed(seed)

        self.args = args
        self.voc12_root = voc12_root
        self.img_name_list_path = 'datasets/' + img_name_list_path
        self.transform = transform
                
        image_dir = os.path.join(self.voc12_root, IMG_FOLDER_NAME)
        target_dir = os.path.join(self.voc12_root, TARGET_FOLDER_NAME)

        """
        with open(os.path.join(self.img_name_list_path), 'r') as f:
            self.file_names = [x.strip() for x in f.readlines()]
        """
        self.file_names = np.loadtxt(self.img_name_list_path, dtype=np.int32)
        self.images = []
        self.targets = []
        for name_int in self.file_names:
            name_str = self._decode_int_filename(name_int)
            self.images.append(os.path.join(image_dir, name_str + '.jpg'))
            self.targets.append(os.path.join(target_dir, name_str + '.png'))
        self.labels = self._load_image_label_list_from_npy()
        
    def __getitem__(self, idx):
        name = self.file_names[idx]
        name_str = decode_int_filename(name)
        img = Image.open(self.images[idx]).convert('RGB')
        target = Image.open(self.targets[idx])
        label = self.labels[idx]
        
        if self.transform is not None:
            img, target = self.transform(img, target)
        
        return {'name': name_str, 'img': img, 'target': target, 'label': label}

    def __len__(self):
        return len(self.images)
    @classmethod
    def decode_target(cls, mask):
        return cls.cmap[mask]
    

    @staticmethod
    def _decode_int_filename(int_filename):
        s = str(int(int_filename))
        return s[:4] + '_' + s[4:]

    def _load_image_label_list_from_npy(self):
        cls_labels_dict = np.load('datasets/voc12/cls_labels.npy', allow_pickle=True).item()
        return np.array([cls_labels_dict[name] for name in self.file_names])
            
