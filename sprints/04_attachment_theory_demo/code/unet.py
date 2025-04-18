import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, List

class ConvBlock(nn.Module):
    """
    Convolutional block: Conv2d -> BatchNorm2d -> ReLU
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))

class UpConvBlock(nn.Module):
    """
    Up-sampling convolutional block: Upsample -> concatenate -> ConvBlock
    """
    def __init__(self, in_channels: int, out_channels: int, mode: str = "nearest") -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode=mode)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # concatenate along channel dimension
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    """
    U-Net architecture for segmentation tasks.

    Args:
        in_channels: Number of input channels (e.g., 3 for RGB).
        out_channels: Number of output channels (e.g., 1 for binary mask).
        features: Sequence of feature sizes for encoder/decoder blocks.
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: Sequence[int] = (64, 128, 256, 512),
    ) -> None:
        super().__init__()
        # Encoder path
        self.downs = nn.ModuleList([])
        prev_channels = in_channels
        for feature in features:
            self.downs.append(ConvBlock(prev_channels, feature))
            prev_channels = feature
        # Bottleneck
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)
        # Decoder path
        self.ups = nn.ModuleList([])
        rev_features = list(reversed(features))
        in_channels_decoder = features[-1] * 2
        for feature in rev_features:
            # After concatenation: in_channels_decoder + feature
            self.ups.append(UpConvBlock(in_channels_decoder + feature, feature))
            in_channels_decoder = feature
        # Final 1x1 convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections: List[torch.Tensor] = []
        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        # Bottleneck
        x = self.bottleneck(x)
        # Reverse skip connections for decoder
        skip_connections = skip_connections[::-1]
        # Decoder
        for idx, up in enumerate(self.ups):
            skip = skip_connections[idx]
            x = up(x, skip)
        # Final segmentation map
        return self.final_conv(x)

if __name__ == "__main__":
    # Smoke test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=1).to(device)
    x = torch.randn(1, 3, 256, 256, device=device)
    with torch.no_grad():
        out = model(x)
    print(f"UNet output shape: {out.shape}") 