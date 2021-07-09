from net.backbone import resnet, vgg
from net.model.deeplab import DeepLabModel, DeepLabV3Head
from net.model.fcn import FCNModel, FCN8s
from net.utils import IntermediateLayerGetter


def _deeplab_with_resnet(arch_type, backbone, num_classes, output_stride, pretrained_backbone):
    if output_stride == 8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]
    
    backbone = resnet.__dict__[backbone](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)
    
    inplanes = 2048
    low_level_planes = 256
    
    if arch_type == 'deeplabv3':
        return_layers = {'layer4': 'out'}
        classifier = DeepLabV3Head(inplanes, num_classes, aspp_dilate)
    
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = DeepLabModel(backbone, classifier)
    
    return model


def _fcn_with_vgg(arch_type, backbone, num_classes, pretrained_backbone):
    backbone = vgg.__dict__[backbone](
        pretrained=pretrained_backbone
    )

    inplanes = 512

    if arch_type == 'fcn8s':
        classifier = FCN8s(inplanes, num_classes)
    
    model = FCNModel(backbone, classifier)

    return model


def _load_model(arch_type, backbone, num_classes, output_stride, pretrained_backbone):
    """Load segmmentation model
    
    Args:
        arch_type (str): deeplabv3
        backbone  (str): Select backbone network
    """
    if arch_type.startswith('deeplab'):
        if backbone.startswith('resnet'):
            model = _deeplab_with_resnet(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)            
        else:
            raise NotImplementedError
    
    elif arch_type.startswith('fcn'):
        if backbone.startswith('vgg'):
            model = _fcn_with_vgg(arch_type, backbone, num_classes, pretrained_backbone)

        else:
            raise NotImplementedError

    return model


def deeplabv3_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.
    
    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'resnet50', num_classes, output_stride, pretrained_backbone=pretrained_backbone)


def deeplabv3_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.
    """
    return _load_model('deeplabv3', 'resnet101', num_classes, output_stride, pretrained_backbone=pretrained_backbone)


def fcn8s_vgg16(num_classes=21, output_stride=None, pretrained_backbone=True):
    """Constructs a FCN8s model with a VGG16
    """
    return _load_model('fcn8s', 'vgg16', num_classes, output_stride=None, pretrained_backbone=pretrained_backbone)