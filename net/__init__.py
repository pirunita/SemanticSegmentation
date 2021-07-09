
from .modeling import *

MODEL_MAP = {
    'deeplabv3_resnet50': deeplabv3_resnet50,
    'deeplabv3_resnet101': deeplabv3_resnet101,
    'fcn8s_vgg16' : fcn8s_vgg16
}