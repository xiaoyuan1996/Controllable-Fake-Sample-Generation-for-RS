import torch
import torch.nn as nn
import torch.nn.init
from .self_resnet import resnet18
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, ):
        super(Discriminator, self).__init__()


        self.resnet = resnet18(pretrained=False)

        self.sigmoid = nn.Sigmoid()

        self.linear = nn.Linear(in_features=512, out_features=2)

    def forward(self, img):
        x1 = self.resnet.conv1(img)
        x2 = self.resnet.bn1(x1)
        x3 = self.resnet.relu(x2)
        x4 = self.resnet.maxpool(x3)

        f1 = self.resnet.layer1(x4)
        f2 = self.resnet.layer2(f1)
        f3 = self.resnet.layer3(f2)
        f4 = self.resnet.layer4(f3)

        # batch * 512
        tmp = f4.shape[0]
        feature = f4.view(tmp, 512, -1)
        mean_f = torch.mean(feature,dim=-1)
        #print(mean_f.shape)
        solo_feature = self.linear(mean_f)

        # torch.Size([10, 192, 64, 64])
        # torch.Size([10, 768, 64, 64])
        # torch.Size([10, 512])
        return self.sigmoid(solo_feature)
