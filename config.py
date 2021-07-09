import argparse

def get_args():
    parser = argparse.ArgumentParser(description='SemanticSegmentation')
    
    parser.add_argument("--name", type=str, default='DeepLab')
    parser.add_argument("--mode", type=str, default='train', choices=['train', 'val', 'test'])
    parser.add_argument("--gpu_id", type=str, default="0,1")
        
    # Dataset Options
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc'], help='dataset name')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None")
    parser.add_argument("--crop_size", type=int, default=448)
    
    parser.add_argument("--train_list", type=str, default="voc12/train_aug.txt")
    parser.add_argument("--val_list", type=str, default="voc12/val.txt")
    parser.add_argument("--test_list", type=str, default="voc12/test.txt")
    
    parser.add_argument("--voc12_root", type=str, default="VOCdevkit/VOC2012",
                        help="path to VOC 2012 Devkit, must contain ./JPEGImages as subdirectory.")

    # Setting
    parser.add_argument("--session", type=int, default=0)
    parser.add_argument("--log_path", type=str, default='logs')
    parser.add_argument("--checkpoint_path", type=str, default='checkpoints')
    parser.add_argument("--training_val_path", type=str, default='training_val')
    parser.add_argument("--val_path", type=str, default='vals')
    parser.add_argument("--checkpoint", type=str, default=None, help="load model to validate")
    
    # Visualizer
    parser.add_argument("--enable_vis", action='store_true', default=True,
                        help='use visdom for visualization')
    parser.add_argument("--vis_port", type=str, default='8097',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    
    # Baseline Options
    parser.add_argument("--based_model", type=str, default='deeplabv3_resnet50',
                        choices=['deeplabv3_wideresnet38',
                                 'deeplabv3_resnet50', 
                                 'deeplabv3_resnet101',
                                 'fcn8s_vgg16'],
                        help="baseline semantic segmentation model")
    
    # Train Hyperparameters
    parser.add_argument("--lr", type=float, default=0.01, 
                        help="learning rate")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate schedular")
    parser.add_argument("--total_epoch", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16,
                        help="batch_size")
    
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--random_seed", type=int, default=1)
    
    return parser.parse_args()
    