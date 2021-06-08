from torch import nn


def encoder(in_channel, out_channel, kernel=4, stride=1, padding=0, leaky_relu_slope=0.2, batch_norm=256):
    conv_layer = nn.Conv2d(in_channel, out_channel, kernel, stride=stride, padding=padding)
    relu_layer = nn.LeakyReLU(negative_slope=leaky_relu_slope)
    batch_norm_layer = nn.BatchNorm2d(batch_norm)
    return nn.Sequential(
        relu_layer,
        conv_layer,
        batch_norm_layer
    )


def encoder_with_dropout(in_channel, out_channel, kernel=4, stride=1, padding=0, leaky_relu_slope=0.2, batch_norm=256):
    conv_layer = nn.Conv2d(in_channel, out_channel, kernel, stride=stride, padding=padding)
    relu_layer = nn.LeakyReLU(negative_slope=leaky_relu_slope)
    batch_norm_layer = nn.BatchNorm2d(batch_norm)
    dropout_layer = nn.Dropout2d()
    return nn.Sequential(
        relu_layer,
        conv_layer,
        batch_norm_layer,
        dropout_layer
    )


def encoder_conv(in_channel, out_channel, kernel=4, stride=1, padding=0):
    conv_layer = nn.Conv2d(in_channel, out_channel, kernel, stride=stride, padding=padding)
    return nn.Sequential(
        conv_layer
    )


def encoder_sig(in_channel, out_channel, kernel=4, stride=1, padding=0, leaky_relu_slope=0.2):
    conv_layer = nn.Conv2d(in_channel, out_channel, kernel, stride=stride, padding=padding)
    relu_layer = nn.LeakyReLU(negative_slope=leaky_relu_slope)
    sigmoid_layer = nn.Sigmoid()
    return nn.Sequential(
        relu_layer,
        conv_layer,
        sigmoid_layer
    )
