from torch import nn


def decoder(in_channel, out_channel, kernel=4, stride=1, padding=0, batch_norm=256):
    conv_layer = nn.ConvTranspose2d(in_channel, out_channel, kernel, stride=stride, padding=padding)
    relu_layer = nn.ReLU()
    batch_norm_layer = nn.BatchNorm2d(batch_norm)
    return nn.Sequential(
        relu_layer,
        conv_layer,
        batch_norm_layer
    )


def decoder_with_dropout(in_channel, out_channel, kernel=4, stride=1, padding=0, batch_norm=256):
    conv_layer = nn.ConvTranspose2d(in_channel, out_channel, kernel, stride=stride, padding=padding)
    relu_layer = nn.ReLU()
    batch_norm_layer = nn.BatchNorm2d(batch_norm)
    dropout_layer = nn.Dropout2d()
    return nn.Sequential(
        relu_layer,
        conv_layer,
        batch_norm_layer,
        dropout_layer
    )


def decoder_conv(in_channel, out_channel, kernel=4, stride=1, padding=0):
    conv_layer = nn.ConvTranspose2d(in_channel, out_channel, kernel, stride=stride, padding=padding)
    relu_layer = nn.ReLU()
    tanh_layer = nn.Tanh()
    return nn.Sequential(
        relu_layer,
        conv_layer,
        tanh_layer
    )
