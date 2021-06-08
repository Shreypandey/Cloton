import torch
from torch import nn

from blocks import encoder, decoder


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = encoder.encoder_conv(25, 64, stride=2, padding=1)
        self.enc2 = encoder.encoder(64, 128, stride=2, padding=1, leaky_relu_slope=0.2, batch_norm=128)
        self.enc3 = encoder.encoder(128, 256, stride=2, padding=1, leaky_relu_slope=0.2, batch_norm=256)
        self.enc4 = encoder.encoder(256, 512, stride=2, padding=1, leaky_relu_slope=0.2, batch_norm=512)
        self.enc5 = encoder.encoder(512, 512, stride=2, padding=1, leaky_relu_slope=0.2, batch_norm=512)
        self.enc6 = encoder.encoder(512, 512, stride=2, padding=1, leaky_relu_slope=0.2, batch_norm=512)
        self.dec6 = decoder.decoder_with_dropout(512, 512, stride=2, padding=1, batch_norm=512)
        self.dec5 = decoder.decoder(512, 512, stride=2, padding=1, batch_norm=512)
        self.dec4 = decoder.decoder(512, 256, stride=2, padding=1, batch_norm=256)
        self.dec3 = decoder.decoder(256, 128, stride=2, padding=1, batch_norm=128)
        self.dec2 = decoder.decoder(128, 64, stride=2, padding=1, batch_norm=64)
        self.dec1 = decoder.decoder_conv(64, 4, stride=2, padding=1)

    def forward(self, x):
        out_enc1 = self.enc1(x)
        # print(out_enc1.size())
        out_enc2 = self.enc2(out_enc1)
        # print(out_enc2.size())
        out_enc3 = self.enc3(out_enc2)
        # print(out_enc3.size())
        out_enc4 = self.enc4(out_enc3)
        # print(out_enc4.size())
        out_enc5 = self.enc5(out_enc4)
        # print(out_enc5.size())
        out_enc6 = self.enc6(out_enc5)
        # print(out_enc6.size())
        out_dec6 = self.dec6(out_enc6)
        # print(out_dec6.size())
        out_dec5 = self.dec5(out_dec6 + out_enc5)
        # print(out_dec5.size())
        out_dec4 = self.dec4(out_dec5 + out_enc4)
        # print(out_dec4.size())
        out_dec3 = self.dec3(out_dec4 + out_enc3)
        # print(out_dec3.size())
        out_dec2 = self.dec2(out_dec3 + out_enc2)
        # print(out_dec2.size())
        out_dec1 = self.dec1(out_dec2 + out_enc1)
        # print(out_dec1.size())
        out = out_dec1
        return out


# generator_model = Generator()
# generator_model = generator_model.cuda()
# input = torch.zeros(1, 25, 256, 192).cuda()
# print(input.size())
# output = generator_model(input)
# print(output)
