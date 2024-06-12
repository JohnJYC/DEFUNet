from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, dropout_prob=0.5, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.dropout_prob = dropout_prob

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DEFDown(64, 128)
        self.down2 = DEFDown(128, 256)
        self.down3 = DEFDown(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = DEFDown(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        # ドロップアウト層の追加
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.dropout(x)
        x = self.up2(x, x3)
        x = self.dropout(x)
        x = self.up3(x, x2)
        x = self.dropout(x)
        x = self.up4(x, x1)
        x = self.dropout(x)

        logits = self.outc(x)
        return logits
