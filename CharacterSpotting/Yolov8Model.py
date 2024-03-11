import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOv8(nn.Module):
    def __init__(self, num_classes=80):
        super(YOLOv8, self).__init__()

        self.conv_layers = nn.ModuleList([
            self._make_conv_layer(3, 32, kernel_size=3, stride=1, padding=1),   # 0
            self._make_conv_layer(32, 64, kernel_size=3, stride=2, padding=1),  # 1
            self._make_conv_layer(64, 32, kernel_size=1, stride=1, padding=1),  # 2
            self._make_conv_layer(32, 64, kernel_size=3, stride=1, padding=1),  # 3
            self._make_conv_layer(64, 128, kernel_size=3, stride=2, padding=1), # 4
            self._make_conv_layer(128, 64, kernel_size=1, stride=1, padding=1), # 5
            self._make_conv_layer(64, 128, kernel_size=3, stride=1, padding=1), # 6
            self._make_conv_layer(128, 64, kernel_size=1, stride=1, padding=1), # 7
            self._make_conv_layer(64, 128, kernel_size=3, stride=1, padding=1), # 8
            self._make_conv_layer(128, 256, kernel_size=3, stride=2, padding=1),# 9
            self._make_conv_layer(256, 128, kernel_size=1, stride=1, padding=1),# 10
            self._make_conv_layer(128, 256, kernel_size=3, stride=1, padding=1),# 11
            self._make_conv_layer(256, 128, kernel_size=1, stride=1, padding=1),# 12
            self._make_conv_layer(128, 256, kernel_size=3, stride=1, padding=1),# 13
            self._make_conv_layer(256, 128, kernel_size=1, stride=1, padding=1),# 14
            self._make_conv_layer(128, 256, kernel_size=3, stride=1, padding=1),# 15
            self._make_conv_layer(256, 128, kernel_size=1, stride=1, padding=1),# 16
            self._make_conv_layer(128, 256, kernel_size=3, stride=1, padding=1),# 17
            self._make_conv_layer(256, 128, kernel_size=1, stride=1, padding=1),# 18
            self._make_conv_layer(128, 256, kernel_size=3, stride=1, padding=1),# 19
            self._make_conv_layer(256, 128, kernel_size=1, stride=1, padding=1),# 20
            self._make_conv_layer(128, 256, kernel_size=3, stride=1, padding=1),# 21
            self._make_conv_layer(256, 512, kernel_size=3, stride=2, padding=1),# 22
            self._make_conv_layer(512, 256, kernel_size=1, stride=1, padding=1),# 23
            self._make_conv_layer(256, 512, kernel_size=3, stride=1, padding=1),# 24
            self._make_conv_layer(512, 256, kernel_size=1, stride=1, padding=1),# 25
            self._make_conv_layer(256, 512, kernel_size=3, stride=1, padding=1),# 26
            self._make_conv_layer(512, 256, kernel_size=1, stride=1, padding=1),# 27
            self._make_conv_layer(256, 512, kernel_size=3, stride=1, padding=1),# 28
            self._make_conv_layer(512, 256, kernel_size=1, stride=1, padding=1),# 29
            self._make_conv_layer(256, 512, kernel_size=3, stride=1, padding=1),# 30
            self._make_conv_layer(512, 256, kernel_size=1, stride=1, padding=1),# 31
            self._make_conv_layer(256, 512, kernel_size=3, stride=1, padding=1),# 32
            self._make_conv_layer(512, 1024, kernel_size=3, stride=2, padding=1),# 33
            self._make_conv_layer(1024, 512, kernel_size=1, stride=1, padding=1),# 34
            self._make_conv_layer(512, 1024, kernel_size=3, stride=1, padding=1),# 35
            self._make_conv_layer(1024, 512, kernel_size=1, stride=1, padding=1),# 36
            self._make_conv_layer(512, 1024, kernel_size=3, stride=1, padding=1),# 37
            self._make_conv_layer(1024, 512, kernel_size=1, stride=1, padding=1),# 38
            self._make_conv_layer(512, 1024, kernel_size=3, stride=1, padding=1),# 39
            self._make_conv_layer(1024, 512, kernel_size=1, stride=1, padding=1),# 40
            self._make_conv_layer(512, 1024, kernel_size=3, stride=1, padding=1),# 41
            self._make_conv_layer(1024, 512, kernel_size=1, stride=1, padding=1),# 42
            self._make_conv_layer(512, 1024, kernel_size=3, stride=1, padding=1),# 43
            self._make_conv_layer(1024, 512, kernel_size=1, stride=1, padding=1),# 44
            self._make_conv_layer(512, 1024, kernel_size=3, stride=1, padding=1),# 45
            self._make_conv_layer(1024, 512, kernel_size=1, stride=1, padding=1),# 46
            self._make_conv_layer(512, 1024, kernel_size=3, stride=1, padding=1),# 47
            self._make_conv_layer(1024, 512, kernel_size=1, stride=1, padding=1),# 48
            self._make_conv_layer(512, 1024, kernel_size=3, stride=1, padding=1),# 49
            self._make_conv_layer(1024, 512, kernel_size=1, stride=1, padding=1),# 50
            self._make_conv_layer(512, 1024, kernel_size=3, stride=1, padding=1),# 51
            self._make_conv_layer(1024, 512, kernel_size=1, stride=1, padding=1),# 52
            self._make_conv_layer(512, 1024, kernel_size=3, stride=1, padding=1),# 53
            self._make_conv_layer(1024, 512, kernel_size=1, stride=1, padding=1),# 54
            self._make_conv_layer(512, 1024, kernel_size=3, stride=1, padding=1),# 55
            self._make_conv_layer(1024, 512, kernel_size=1, stride=1, padding=1),# 56
            self._make_conv_layer(512, 1024, kernel_size=3, stride=1, padding=1),# 57
            self._make_conv_layer(1024, 512, kernel_size=1, stride=1, padding=1),# 58
            self._make_conv_layer(512, 1024, kernel_size=3, stride=1, padding=1),# 59
            self._make_conv_layer(1024, 512, kernel_size=1, stride=1, padding=1),# 60
            self._make_conv_layer(512, 1024, kernel_size=3, stride=1, padding=1),# 61
            self._make_conv_layer(1024, 512, kernel_size=1, stride=1, padding=1),# 62
            self._make_conv_layer(512, 1024, kernel_size=3, stride=1, padding=1),# 63
            self._make_conv_layer(1024, 512, kernel_size=1, stride=1, padding=1),# 64
            self._make_conv_layer(512, 1024, kernel_size=3, stride=1, padding=1),# 65
            self._make_conv_layer(1024, 255, kernel_size=1, stride=1, padding=1),# 66
            self._make_conv_layer(512, 256, kernel_size=1, stride=1, padding=1),# 67
            self._make_upsample_layer(scale_factor=2),                           # 68
            self._make_route_layer(layers=[-1, 61]),                              # 69
            self._make_conv_layer(512 + 256, 256, kernel_size=1, stride=1, padding=1),# 70
            self._make_conv_layer(256, 512, kernel_size=3, stride=1, padding=1),# 71
            self._make_conv_layer(512, 256, kernel_size=1, stride=1, padding=1),# 72
            self._make_conv_layer(256, 512, kernel_size=3, stride=1, padding=1),# 73
            self._make_conv_layer(512, 256, kernel_size=1, stride=1, padding=1),# 74
            self._make_conv_layer(256, 512, kernel_size=3, stride=1, padding=1),# 75
            self._make_conv_layer(512, 255, kernel_size=1, stride=1, padding=1),# 76
            self._make_route_layer(layers=[-1, 36]),                              # 77
            self._make_conv_layer(256 + 128, 128, kernel_size=1, stride=1, padding=1),# 78
            self._make_upsample_layer(scale_factor=2),                           # 79
            self._make_route_layer(layers=[-1, 19]),                              # 80
            self._make_conv_layer(256 + 128, 256, kernel_size=1, stride=1, padding=1),# 81
            self._make_conv_layer(256, 512, kernel_size=3, stride=1, padding=1),# 82
            self._make_conv_layer(512, 256, kernel_size=1, stride=1, padding=1),# 83
            self._make_conv_layer(256, 512, kernel_size=3, stride=1, padding=1),# 84
            self._make_conv_layer(512, 256, kernel_size=1, stride=1, padding=1),# 85
            self._make_conv_layer(256, 512, kernel_size=3, stride=1, padding=1),# 86
            self._make_conv_layer(512, 255, kernel_size=1, stride=1, padding=1),# 87
        ])

    def forward(self, x):
        outputs = []
        route_layers = []

        for i, layer in enumerate(self.conv_layers):
            if isinstance(layer, nn.Conv2d):
                x = layer(x)
            elif isinstance(layer, nn.BatchNorm2d):
                x = layer(x)
            elif isinstance(layer, nn.LeakyReLU):
                x = layer(x)
            elif isinstance(layer, RouteLayer):
                layers = [int(l) for l in layer.layers.split(',')]
                if len(layers) == 1:
                    x = route_layers[layers[0]]
                else:
                    concatenated_layers = [route_layers[l] for l in layers]
                    x = torch.cat(concatenated_layers, 1)
            elif isinstance(layer, UpsampleLayer):
                x = F.interpolate(x, scale_factor=layer.scale_factor, mode='nearest')
            outputs.append(x)
            route_layers.append(x)

        return outputs

    def _make_conv_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def _make_route_layer(self, layers):
        return RouteLayer(layers)

    def _make_upsample_layer(self, scale_factor):
        return UpsampleLayer(scale_factor)


class RouteLayer(nn.Module):
    def __init__(self, layers):
        super(RouteLayer, self).__init__()
        self.layers = layers

    def forward(self, x):
        pass

class UpsampleLayer(nn.Module):
    def __init__(self, scale_factor):
        super(UpsampleLayer, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        pass
