import math
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.prelu(out)
        out = self.conv2(out)
        return x + out


class ELSR(nn.Module):
    def __init__(self, upscale_factor):
        super(ELSR, self).__init__()
        self.layer1 = nn.Conv2d(3, 6, kernel_size=3, padding=1),
        self.layer2_4 = ResBlock(6, 6)
        self.layer5 = nn.Conv2d(6, 3 * (upscale_factor ** 2), kernel_size=3, padding=1),     # 6 -> 48
        self.layer6 = nn.PixelShuffle(upscale_factor)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.out_channels == 48:
                    nn.init.normal_(m.weight.data, mean=0.0, std=0.001)
                    nn.init.zeros_(m.bias.data)
                else:
                    nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2_4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x
