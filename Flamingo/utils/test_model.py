import torch 
from torch import nn


# Your Neural Network
class ConvBlock(nn.Module):

    def __init__(self, num_channels: int, layer_scale_init: float = 1e-6):
        super().__init__()
        self.residual = nn.Sequential(
            nn.GroupNorm(1, num_channels),  # LayerNorm
            nn.GELU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, num_channels),  # LayerNorm
            nn.GELU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
        )
        self.layer_scale = nn.Parameter(torch.tensor(layer_scale_init))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        h = inputs + self.layer_scale * self.residual(inputs)
        return h


class MyClassifier(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, ch_multi: int = 32):
        super().__init__()

        self.stage1 = nn.Sequential(
            # downscale
            nn.Sequential(
                nn.Conv2d(in_channels, ch_multi, kernel_size=2, stride=2, padding=0),
                nn.GroupNorm(1, ch_multi),  # LayerNorm
            ),
            ConvBlock(ch_multi),
            ConvBlock(ch_multi),
        )

        self.stage2 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(ch_multi, 2 * ch_multi, kernel_size=2, stride=2, padding=0),
                nn.GroupNorm(1, 2 * ch_multi),  # LayerNorm
            ),
            ConvBlock(2 * ch_multi),
            ConvBlock(2 * ch_multi),
        )

        self.stage3 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(2 * ch_multi, 4 * ch_multi, kernel_size=2, stride=2, padding=0),
                nn.GroupNorm(1, 4 * ch_multi),  # LayerNorm
            ),
            ConvBlock(4 * ch_multi),
            ConvBlock(4 * ch_multi),
        )

        self.stage4 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(4 * ch_multi, 8 * ch_multi, kernel_size=2, stride=2, padding=0),
                nn.GroupNorm(1, 8 * ch_multi),  # LayerNorm
            ),
            ConvBlock(8 * ch_multi),
            ConvBlock(8 * ch_multi),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.Linear(8 * ch_multi, out_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        h = self.stage1(inputs)
        h = self.stage2(h)
        h = self.stage3(h)
        h = self.stage4(h)
        h = self.classifier(h)
        return h

if __name__ == "__main__":
    model = MyClassifier(3, 100, ch_multi=128)