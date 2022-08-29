from torch import nn
import torch.nn.functional as F
from torchvision import models


class ResNet50Mod(nn.Module):
    def __init__(self):
        super(ResNet50Mod, self).__init__()
        model_resnet50 = models.resnet50(pretrained=True)
        self.freezed_rn50 = nn.Sequential(
            model_resnet50.conv1,
            model_resnet50.bn1,
            model_resnet50.relu,
            model_resnet50.maxpool,
            model_resnet50.layer1,
            model_resnet50.layer2,
            model_resnet50.layer3,
        )
        self.layer4 = model_resnet50.layer4
        self.avgpool = model_resnet50.avgpool
        self.__in_features = model_resnet50.fc.in_features
    
    def forward(self, x):
        x = self.freezed_rn50(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels=3, h=256, dropout=0.5, srcTrain=False):
        super(Encoder, self).__init__()
        
        rnMod = ResNet50Mod()
        self.feature_extractor = rnMod.freezed_rn50
        self.layer4 = rnMod.layer4
        self.avgpool = rnMod.avgpool
        if srcTrain:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = x.expand(x.data.shape[0], 3, 224, 224)
        x = self.feature_extractor(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class Classifier(nn.Module):
    def __init__(self, n_classes, dropout=0.5):
        super(Classifier, self).__init__()
        self.l1 = nn.Linear(2048, n_classes)

    def forward(self, x):
        x = self.l1(x)
        return x


class CNN(nn.Module):
    def __init__(self, in_channels=3, n_classes=31, target=False, srcTrain=False):
        super(CNN, self).__init__()
        self.encoder = Encoder(in_channels=in_channels, srcTrain=srcTrain)
        self.classifier = Classifier(n_classes)
        if target:
            for param in self.classifier.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, h=500, args=None):
        super(Discriminator, self).__init__()
        self.l1 = nn.Linear(2048, h)
        self.l2 = nn.Linear(h, h)
        self.l3 = nn.Linear(h, 2)
        self.l4 = nn.LogSoftmax(dim=1)
        self.slope = args.slope

    def forward(self, x):
        x = F.leaky_relu(self.l1(x), self.slope)
        x = F.leaky_relu(self.l2(x), self.slope)
        x = self.l3(x)
        x = self.l4(x)
        return x
