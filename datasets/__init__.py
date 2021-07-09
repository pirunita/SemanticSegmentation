from datasets import transforms as tf
from datasets.voc12.voc import *

def get_dataset(args):
    train_transform = tf.Compose([
        tf.RandomScale((0.9, 1.0)),
        tf.RandomCrop(size=(args.crop_size, args.crop_size), pad_if_needed=True),
        tf.RandomHorizontalFlip(),
        tf.ToTensor(),
        tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    #deeplab
    """
    val_transform = tf.Compose([
        tf.ToTensor(),
        tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    """
    val_transform = tf.Compose([
        tf.Resize(args.crop_size),
        tf.ToTensor(),
        tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    
    if args.dataset.lower() == 'voc':
        train_dataset = VOCSegmentation(args=args, voc12_root=args.voc12_root, \
                        img_name_list_path=args.train_list, transform=train_transform, seed=args.random_seed)
        
        val_dataset = VOCSegmentation(args=args, voc12_root=args.voc12_root, \
                        img_name_list_path=args.val_list, transform=val_transform, seed=args.random_seed)
    
    
    return train_dataset, val_dataset