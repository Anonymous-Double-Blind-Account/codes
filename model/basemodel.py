import torch.nn as nn
from model.resnet import *
from timm.models.vgg import vgg16_bn

class Mlp(nn.Module):
    """
    MLP block including droput rate=0.1 for uncertainty evaluation (MC Dropout)
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CNNEncoder(nn.Module):
    """
    encoder for CNN, including ResNet blocks
    """
    def __init__(self, input_ch, dim=128, depth=3):
        super().__init__()
        self.depth = depth

        self.blocks = nn.ModuleList()
        input_dim = input_ch
        for i in range(self.depth):
            conv = nn.Conv2d(input_dim, dim*2**i, kernel_size=3, padding=1) 
            blk = BasicBlock(dim*2**i, dim*2**i)
            downsample = nn.MaxPool2d(2, 2)

            self.blocks.append(conv)
            self.blocks.append(blk)
            self.blocks.append(downsample)
            input_dim = dim*2**i

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        return x

class vggEncoder(nn.Module):
    """
    image encoder with pretrained VGG16
    """
    def __init__(self, input_ch, dim=128, depth=3):
        super().__init__()
        self.vgg = vgg16_bn(pretrained=True, features_only=True)

    def forward(self, x):
        x = self.vgg(x)[-1]
        return x
