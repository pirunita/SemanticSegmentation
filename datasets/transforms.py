import collections
import numbers
import random

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F

from PIL import Image

class Compose:
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, image, label):

        result = (image, label)
        for t in self.transform:
            result = t(*result)
        
        image, label = result
        return image, label
    

class RandomScale(object):
    def __init__(self, scale_range, interpolation=Image.BILINEAR):
        self.scale_range = scale_range
        self.interpolation = interpolation
    
    def __call__(self, image, label):
        
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        target_size = (int(image.size[1]*scale), int(image.size[0]*scale))

        if label is not None:
            return F.resize(image, target_size, self.interpolation), F.resize(label, target_size, Image.NEAREST)
        else:
            return F.resize(image, target_size, self.interpolation), None
        

class RandomCrop(object):
    """Crop the given PIL Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is made.
        padding (sequence or int, optional): Optional padding on each border of the image.
            Default is 0, i.e no padding. If a sequence of length 4 is provided. It is used to
            pad left, top, right, bottom borders respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the desired size to
            avoid raising an exception.
    """

    def __init__(self, size, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
    
    @staticmethod
    def get_params(image, output_size):
        """Get parameters to crop for random crop.
        Args:
            image (PIL Image)   : Image to be cropped.
            output_size (tuple) : Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to crop for random crop
        """
        w, h = image.size
        th, tw = output_size

        if w == tw and h == th:
            return 0, 0, h, w
        
        i = random.randint(0, h-th)
        j = random.randint(0, w-tw)
        return i, j, th, tw
    
    def __call__(self, image, label):
        """
        Args:
            image (PIL Image): Image to be cropped.
            label (PIL Image): Label to be cropped.
        Returns:
            PIL Image: Cropped image.
            PIL Image: Cropped label.
        """

        if label is not None:
            if self.padding > 0:
                image = F.pad(image, self.padding)
                label = F.pad(label, self.padding)
            
            # pad the width if needed
            if self.pad_if_needed and image.size[0] < self.size[1]:
                image = F.pad(image, padding=int((1 + self.size[1] - image.size[0]) / 2))
                label = F.pad(label, padding=int((1 + self.size[1] - label.size[0]) / 2))
            
            # pad the height if needed
            if self.pad_if_needed and image.size[1] < self.size[0]:
                image = F.pad(image, padding=int((1 + self.size[0] - image.size[1]) / 2))
                label = F.pad(label, padding=int((1 + self.size[0] - label.size[1]) / 2))

            i, j, h, w = self.get_params(image, self.size)

            return F.crop(image, i, j, h, w), F.crop(label, i, j, h, w)

        else:
            if self.padding > 0:
                image = F.pad(image, self.padding)

            # pad the width if needed
            if self.pad_if_needed and image.size[0] < self.size[1]:
                image = F.pad(image, padding=int((1 + self.size[1] - image.size[0]) / 2))
            
            # pad the height if needed
            if self.pad_if_needed and image.size[1] < self.size[0]:
                image = F.pad(image, padding=int((1 + self.size[0] - image.size[1]) / 2))
                
            i, j, h, w = self.get_params(image, self.size)

            return F.crop(image, i, j, h, w), None


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, label):
        """
        Args:
            image (PIL Image): Image to be flipped.
        
        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            if label is not None:
                return F.hflip(image), F.hflip(label)
            else:
                return F.hflip(image), None
        return image, label


class Resize(object):
    """Resize the input PIL Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation
    
    def __call__(self, img, lbl):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        return F.resize(img, self.size, self.interpolation), F.resize(lbl, self.size, Image.NEAREST)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str) 

        
class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    def __init__(self, normalize=True, target_type='uint8'):
        self.normalize = normalize
        self.target_type = target_type
    
    def __call__(self, image, label):
        """
        Note that labels will not be normalized to [0, 1].
        args:
            image (PIL Image or numpy.ndarray): Image to be converted to tensor.
            label (PIL Image or numpy.ndarray): Label to be converted to tensor.
        Returns:
            Tensor: Converted image and label
        """
        if self.normalize:
            if label is not None:
                return F.to_tensor(image), torch.from_numpy(np.array(label, dtype=self.target_type))
            else:
                return F.to_tensor(image), None
        else:
            if label is not None:
                return torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1)), torch.from_numpy(np.array(label, dtype=self.target_type))
            else:
                return torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1)), None


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std  

    def __call__(self, tensor, label):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            tensor (Tensor): Tensor of label. A dummy input for ExtCompose
        Returns:
            Tensor: Normalized Tensor image.
            Tensor: Unchanged Tensor label
        """
        return F.normalize(tensor, self.mean, self.std), label

