import torch
from torch import nn

from blocks import encoder


class Discriminator(nn.Module):
    # initializers
    def __init__(self):
        super().__init__()
        self.conv1 = encoder.encoder_conv(6, 64, kernel=4, stride=2, padding=1)
        self.conv2 = encoder.encoder(64, 128, kernel=4, stride=2, padding=1, batch_norm=128)
        self.conv3 = encoder.encoder(128, 256, kernel=4, stride=2, padding=1, batch_norm=256)
        self.conv4 = encoder.encoder(256, 512, kernel=4, stride=1, padding=1, batch_norm=512)
        self.conv5 = encoder.encoder_sig(512, 1, kernel=4, stride=1, padding=1)

    # forward method
    def forward(self, image, cloth):
        x = torch.cat([image, cloth], 1)
        out1 = self.conv1(x)
        # print(out1.size())
        out2 = self.conv2(out1)
        # print(out2.size())
        out3 = self.conv3(out2)
        # print(out3.size())
        out4 = self.conv4(out3)
        # print(out4.size())
        out5 = self.conv5(out4)
        # print(out5.size())
        out = out5
        return out


# discriminator_model = Discriminator()
# discriminator_model = discriminator_model.cuda()
# input = torch.zeros(1, 3, 256, 192).cuda()
# label = torch.zeros(1, 3, 256, 192).cuda()
# print(input.size())
# print(label.size())
# output = discriminator_model(input, label)
# print(output)
