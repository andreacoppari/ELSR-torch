import math
from torch import nn


class ELSR(nn.Module):
    def __init__(self, upscale_factor):
        super(ELSR, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=3, padding=3//2),
            nn.ReLU(),
            nn.Conv2d(6, 6, kernel_size=3, padding=3//2),
            nn.PReLU(),
            nn.Conv2d(6, 6, kernel_size=3, padding=3//2),
            nn.ReLU(),
        )
        self.last_part = nn.Sequential(
            nn.Conv2d(6, 3 * (upscale_factor ** 2), kernel_size=3, padding=3 // 2),     # 6 -> 48
            nn.PixelShuffle(upscale_factor)
        )

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
        x = self.first_part(x)
        x = self.last_part(x)
        return x


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count