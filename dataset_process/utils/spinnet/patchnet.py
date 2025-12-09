import torch.nn as nn
import torch
from .utils import common


class BaseNet(nn.Module):

    def forward_one(self, x):
        raise NotImplementedError()

    def forward(self, imgs):
        res = self.forward_one(imgs)
        return res


class Cyclindrical_ConvNet(BaseNet):
    def __init__(self, inchan=3, dilated=True, dilation=1, bn=True, bn_affine=False):
        BaseNet.__init__(self)
        self.inchan = inchan
        self.curchan = inchan
        self.dilated = dilated
        self.dilation = dilation
        self.bn = bn
        self.bn_affine = bn_affine
        self.ops = nn.ModuleList([])

    def _make_bn_2d(self, outd):
        return nn.BatchNorm2d(outd, affine=self.bn_affine)

    def _make_bn_3d(self, outd):
        return nn.BatchNorm3d(outd, affine=self.bn_affine)

    def _add_conv_2d(self, outd, k=3, stride=1, dilation=1, bn=True, relu=True):
        d = self.dilation * dilation
        self.dilation *= stride
        self.ops.append(nn.Conv2d(self.curchan, outd, kernel_size=(k, k), dilation=d))
        if bn and self.bn: self.ops.append( self._make_bn_2d(outd) )
        if relu: self.ops.append( nn.ReLU(inplace=True) )
        self.curchan = outd

    def _add_conv_3d(self, outd, k, stride=1, dilation=1, bn=True, relu=True):
        d = self.dilation * dilation
        self.dilation *= stride
        self.ops.append(nn.Conv3d(self.curchan, outd, kernel_size=(k[0], k[1], k[2]), dilation=d))
        if bn and self.bn: self.ops.append( self._make_bn_3d(outd) )
        if relu: self.ops.append( nn.ReLU(inplace=True) )
        self.curchan = outd

    def forward_one(self, x):
        assert self.ops, "You need to add convolutions first"
        for n,op in enumerate(self.ops):
            k_exist = hasattr(op, 'kernel_size')
            if k_exist:
                if len(op.kernel_size) == 3:
                    x = common.pad_image_3d(x, op.kernel_size[1] + (op.kernel_size[1]-1)*(op.dilation[0]-1))
                else:
                    if len(x.shape) == 5:
                        x = x.squeeze(2)
                        mid_feat = x
                    x = common.pad_image(x, op.kernel_size[0] + (op.kernel_size[0]-1)*(op.dilation[0]-1))
            x = op(x)
        try:
            mid_feat
        except NameError:
            return x
        else:
            return x, mid_feat
class Cylindrical_Net (Cyclindrical_ConvNet):
    """
    Compute a 32D descriptor for cylindrical feature maps
    """
    def __init__(self, inchan=16, dim=32, **kw ):
        Cyclindrical_ConvNet.__init__(self, inchan=inchan, **kw)
        add_conv_2d = lambda n, **kw: self._add_conv_2d(n, **kw)
        add_conv_3d = lambda n, **kw: self._add_conv_3d(n, **kw)
        add_conv_3d(64, k=[3, 3, 3])
        add_conv_2d(64)
        add_conv_2d(128)
        add_conv_2d(128)
        add_conv_2d(64)
        add_conv_2d(64)
        add_conv_2d(32)
        add_conv_2d(32, bn=False, relu=False)
        self.out_dim = dim

class Cylindrical_UNet(nn.Module):
    """
    Compute a 32D descriptor for cylindrical feature maps with U-Net-like architecture
    """
    def __init__(self, inchan=16, dim=32):
        super(Cylindrical_UNet, self).__init__()
        
       # Initial Conv3D Block
        self.conv3d = nn.Sequential(
            nn.Conv3d(inchan, 32, kernel_size=(3, 3, 3), stride=1, dilation=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        # U-Net Encoder
        self.encoder1 = self.make_conv_block(32, 32)  # Encoder Level 1
        self.encoder2 = self.make_conv_block(32, 64)  # Encoder Level 2
        self.encoder3 = self.make_conv_block(64, 128)  # Encoder Level 3

        # U-Net Bottleneck
        self.bottleneck = self.make_conv_block(128, 128)

        # U-Net Decoder
        self.decoder3 = self.make_conv_block(128 + 128, 64)  # Concat with Encoder Level 3
        self.decoder2 = self.make_conv_block(64 + 64, 32)   # Concat with Encoder Level 2
        self.decoder1 = self.make_conv_block(32 + 32, 32)   # Concat with Encoder Level 1

        # Final Output Layer
        self.output_layer = nn.Sequential(
            nn.Conv2d(32, dim, kernel_size=3, stride=1, dilation=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        
    def make_conv_block(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bn=True, relu=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation)]
        if bn:
            layers.append(nn.BatchNorm2d(out_channels))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        
        # Conv3D Feature Extraction
        x = self.conv3d(common.pad_image_3d(x, kernel_size=3))
        x = x.squeeze(2)  # Squeeze 3D output to 2D

        # U-Net Encoder
        enc1 = self.encoder1(common.pad_image(x, kernel_size=3))  # Level 1
        enc2 = self.encoder2(common.pad_image(enc1, kernel_size=3))  # Level 2
        enc3 = self.encoder3(common.pad_image(enc2, kernel_size=3))  # Level 3

        # U-Net Bottleneck
        bottleneck = self.bottleneck(common.pad_image(enc3, kernel_size=3))

        # U-Net Decoder with Concatenation-based Skip Connections
        dec3 = self.decoder3(common.pad_image(torch.cat([bottleneck, enc3], dim=1), kernel_size=3))  # Concat with Encoder Level 3
        dec2 = self.decoder2(common.pad_image(torch.cat([dec3, enc2], dim=1), kernel_size=3))  # Concat with Encoder Level 2
        dec1 = self.decoder1(common.pad_image(torch.cat([dec2, enc1], dim=1), kernel_size=3))  # Concat with Encoder Level 1

        # Final Output
        output = self.output_layer(common.pad_image(dec1, kernel_size=3))
        return output, None

class CostBlock(BaseNet):
    def __init__(self, inchan=32, dilated=True, dilation=1, bn=True, bn_affine=False):
        BaseNet.__init__(self)
        self.inchan = inchan
        self.curchan = inchan
        self.dilated = dilated
        self.dilation = dilation
        self.bn = bn
        self.bn_affine = bn_affine
        self.ops = nn.ModuleList([])

    def _make_bn_2d(self, outd):
        return nn.BatchNorm2d(outd, affine=self.bn_affine)

    def _make_bn_3d(self, outd):
        return nn.BatchNorm3d(outd, affine=self.bn_affine)

    def _add_conv_2d(self, outd, k=3, stride=1, dilation=1, bn=True, relu=True):
        d = self.dilation * dilation
        self.dilation *= stride
        self.ops.append(nn.Conv2d(self.curchan, outd, kernel_size=(k, k), dilation=d))
        if bn and self.bn: self.ops.append( self._make_bn_2d(outd) )
        if relu: self.ops.append( nn.ReLU(inplace=True) )
        self.curchan = outd

    def _add_conv_3d(self, outd, k, stride=1, dilation=1, bn=True, relu=True):
        d = self.dilation * dilation
        self.dilation *= stride
        self.ops.append(nn.Conv3d(self.curchan, outd, kernel_size=(k[0], k[1], k[2]), dilation=d))
        if bn and self.bn: self.ops.append( self._make_bn_3d(outd) )
        if relu: self.ops.append( nn.ReLU(inplace=True) )
        self.curchan = outd

    def forward_one(self, x):
        assert self.ops, "You need to add convolutions first"
        for n,op in enumerate(self.ops):
            x = op(x)

        return x


class CostNet(CostBlock):
    """
    Cost aggregation
    """
    def __init__(self, inchan=32, dim=1, **kw ):
        CostBlock.__init__(self, inchan=inchan, **kw)
        add_conv_2d = lambda n, **kw: self._add_conv_2d(n, **kw)
        add_conv_3d = lambda n, **kw: self._add_conv_3d(n, **kw)
        add_conv_3d(32, k=[3, 3, 3])
        add_conv_3d(64, k=[3, 3, 3])
        add_conv_3d(64, k=[3, 1, 3])
        add_conv_3d(128, k=[3, 1, 3])
        add_conv_3d(128, k=[3, 1, 3])
        add_conv_3d(64, k=[3, 1, 3])
        add_conv_3d(64, k=[3, 1, 3])
        add_conv_3d(32, k=[3, 1, 3])
        add_conv_3d(32, k=[3, 1, 3])
        add_conv_3d(dim, k=[2, 1, 2], bn=False, relu=False)
        self.out_dim = dim

