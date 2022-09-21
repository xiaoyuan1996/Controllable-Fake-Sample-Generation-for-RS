import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, num_classes = 2):
        super().__init__()
        self.num_classes = num_classes

        net = []
        # 1:预先定义
        channels_in = [3, 64, 128, 256, 512]
        channels_out = [64, 128, 256, 512, 1]
        padding = [1, 1, 1, 1, 0]
        active = ["LR", "LR", "LR", "LR", "sigmoid"]
        for i in range(len(channels_in)):
            net.append(nn.Conv2d(in_channels=channels_in[i], out_channels=channels_out[i],
                                 kernel_size=4, stride=2, padding=padding[i], bias=False))
            if i == 0:
                net.append(nn.LeakyReLU(0.2))
            elif active[i] == "LR":
                net.append(nn.BatchNorm2d(num_features=channels_out[i]))
                net.append(nn.LeakyReLU(0.2))
            elif active[i] == "sigmoid":
                net.append(nn.Sigmoid())
        self.discriminator = nn.Sequential(*net)

    def forward(self, x):
        # label = label.unsqueeze(2).unsqueeze(3)
        # label = label.repeat(1, 1, x.size(2), x.size(3))
        # data = torch.cat(tensors=(x, label), dim=1)
        print(x.shape)
        out = self.discriminator(x)
        out = out.view(x.size(0), -1)
        return out
#device='cuda' if torch.cuda.is_available() else 'cpu'
