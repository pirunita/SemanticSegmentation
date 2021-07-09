import torch
import torch.nn as nn
import torch.nn.functional as F

class FCNModel(nn.Module):
    def __init__(self, backbone, classifier):
        super(FCNModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(x, features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)

        return x
    

class FCN8s(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FCN8s, self).__init__()

        # fc6
        self.fc6 = nn.Conv2d(in_channels, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_pool5 = nn.Conv2d(4096, num_classes, 1)
        self.score_pool4 = nn.Conv2d(512, num_classes, 1)
        self.score_pool3 = nn.Conv2d(256, num_classes, 1)

        # Transpose Convolution
        self.upscore5 = nn.ConvTranspose2d(
            num_classes, num_classes, 4, stride=2)
        self.upscore4 = nn.ConvTranspose2d(
            num_classes, num_classes, 13, stride=2)
        self.upscore3 = nn.ConvTranspose2d(
            num_classes, num_classes, 4, stride=2)

        self._init_weight()

    def forward(self, x, features):

        x3 = features['x3']
        x4 = features['x4']
        x5 = features['x5']

        x = self.fc6(x5)
        x = self.relu6(x)
        x = self.drop6(x)

        x = self.fc7(x)
        x = self.relu7(x)
        x = self.drop7(x)
        #import pdb; pdb.set_trace()
        # Predict classificaiton on x5
        x = self.score_pool5(x)             # (b, 21, 4, 4)

        # Transpose Convlution from x5
        x = self.upscore5(x)    
        upscore5 = x                        # (b, 21, 10, 10)

        # Predict classificaiton on x4
        x = self.score_pool4(x4 * 0.01)     # (b, 21, 28, 28)
        x = x[:,:,5:5+upscore5.size()[2], 5:5+upscore5.size()[3]].contiguous()
        
        # Skip Connection
        x = x + upscore5                    # (b, 21, 18, 18)

        # Transpose Convolution from (x4 + x5)
        x = self.upscore4(x)                # (b, 21, 152, 152)
        upscore4 = x

        # Predict classification on x3
        x = self.score_pool3(x3 * 0.0001)   # (b, 21, 56, 56)
        x = x[:,:,9:9+upscore4.size()[2], 9:9+upscore4.size()[3]].contiguous()
        
        # Skip Connection
        x = x + upscore4
        
        # Transpose Convolution from (x3 + (x4 + x5))
        x = self.upscore3(x)
        
        return x
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



